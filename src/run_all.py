"""Run the complete analysis pipeline (revision v2).

Environment variables:
    DATA_ROOT       folder containing the PSWCI cohort folders (raw SPSS files)
    RTW_OUT_DIR     where aggregated tables/figures are written
    RTW_CACHE_DIR   cache for fitted models / row-level predictions (never share)
    RTW_USE_PARQUET 1 = read DATA_ROOT/00_derived_data/pswci_harmonized_long.parquet if present
    RTW_STAGES      comma-separated subset of stages (default: all)
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtw_config import B_BOOT, B_SHIFT, N_JOBS, R_UPDATE, SEED, paths  # noqa: E402
from rtw_data import build_analytic, load_long, task_frame  # noqa: E402
import rtw_analyses as A  # noqa: E402
import rtw_shift as SH  # noqa: E402
import rtw_update as U  # noqa: E402
from rtw_develop import designs_for, develop, params_table  # noqa: E402
from rtw_fair import fairness  # noqa: E402
from rtw_models import ONEHOT_MODELS  # noqa: E402

TASKS = ("A1", "A2", "P")
SHIFT_MODELS = ["LR", "RidgeOH", "XGB", "LGBM", "CatBoost", "EBM"]
SECONDARY = [("c1to3", (1,), 3, SHIFT_MODELS), ("c2to3", (2,), 3, SHIFT_MODELS),
             ("chain12", (1,), 2, ["LR", "RidgeOH", "XGB"]), ("loco1", (2, 3), 1, ["LR", "RidgeOH", "XGB"]),
             ("loco2", (1, 3), 2, ["LR", "RidgeOH", "XGB"])]
DESIGN_LABEL = {"primary": "C1+C2 -> C3", "c1to3": "C1 -> C3", "c2to3": "C2 -> C3", "chain12": "C1 -> C2",
                "loco1": "C2+C3 -> C1", "loco2": "C1+C3 -> C2"}


def main():
    np.random.seed(SEED)
    data_root, out, cache = paths()
    stages = os.environ.get("RTW_STAGES", "all").split(",")
    run = lambda s: "all" in stages or s in stages  # noqa: E731
    logf = open(os.path.join(out, "run_log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    log(f"=== v2 run start; stages={stages}; B_BOOT={B_BOOT} B_SHIFT={B_SHIFT} R_UPDATE={R_UPDATE} N_JOBS={N_JOBS}")
    a = build_analytic(load_long(data_root))
    T = lambda n: os.path.join(out, "tables", n)  # noqa: E731
    if run("describe"):
        A.descriptives(a, out)
        A.predictor_table(a, out)
        SH.prevalence_table(a).to_csv(T("shift_prevalence.csv"), index=False)
    res = {t: develop(a, t, (1, 2), 3, tag="primary", cache_dir=cache, log=log) for t in TASKS}
    pd.concat([params_table(r) for r in res.values()]).to_csv(T("tuning_primary.csv"), index=False)
    log(f"--- primary development done ({time.time() - t0:.0f}s)")
    if run("evaluate"):
        A.riley(res, a, out)
        for t in TASKS:
            A.evaluate_task(res[t], a, out, log=log)
        log(f"--- evaluation done ({time.time() - t0:.0f}s)")
    sec = {}
    for tag, trc, tec, models in SECONDARY:
        for t in TASKS:
            sec[(tag, t)] = develop(a, t, trc, tec, reduced=True, tag=tag, cache_dir=cache, log=log, models=models,
                                    select_repeats=1)
    log(f"--- secondary development done ({time.time() - t0:.0f}s)")
    if run("secondary"):
        rows = []
        for (tag, t), r in sec.items():
            rows.append(A.quick_eval(r["y_test"], A._ordered(r), t, B_BOOT, DESIGN_LABEL[tag],
                                     extra={"design": tag, "primary_in_design": r["primary"]}))
        for t in TASKS:
            rows.append(A.quick_eval(res[t]["y_test"], A._ordered(res[t]), t, B_BOOT, DESIGN_LABEL["primary"],
                                     extra={"design": "primary", "primary_in_design": res[t]["primary"]}))
        pd.concat(rows, ignore_index=True).to_csv(T("secondary_designs.csv"), index=False)
        pd.concat([params_table(r, repeats=1) for r in sec.values()]).to_csv(T("tuning_secondary.csv"), index=False)
        log(f"--- secondary evaluation done ({time.time() - t0:.0f}s)")
    if run("shift"):
        ce = pd.concat([SH.cohort_effects(a, t) for t in TASKS])
        ce.to_csv(T("shift_cohort_effects.csv"), index=False)
        it = pd.DataFrame([SH.outcome_in_domain_test(a, t, s) for t in TASKS for s in ((1,), (2,), (1, 2))])
        it.to_csv(T("shift_outcome_in_domain_test.csv"), index=False)
        pairs = [((1,), (3,)), ((2,), (3,)), ((1, 2), (3,)), ((1,), (2,))]
        adv, drv = SH.adversarial(a, pairs, B=B_BOOT)
        adv2, drv2 = SH.adversarial(a, pairs, B=B_BOOT, restricted=True)
        pd.concat([adv, adv2]).to_csv(T("shift_adversarial.csv"), index=False)
        pd.concat([drv, drv2]).to_csv(T("shift_adversarial_drivers.csv"), index=False)
        dis, info, cm, iws = [], [], [], []
        for t in TASKS:
            d = task_frame(a, t)
            D = designs_for(d, t)
            for tag, r in (("primary", res[t]), ("c1to3", sec[("c1to3", t)]), ("c2to3", sec[("c2to3", t)])):
                Xn = D["num"][0]
                Xs, Xt = Xn.loc[r["dev_index"]], Xn.loc[r["test_index"]]
                models = [m for m in ("RidgeOH", "XGB", res[t]["primary"]) if m in r["oof"]]
                models = list(dict.fromkeys(models))
                tab, inf = SH.disde(Xs, r["y_dev"], {m: r["oof"][m] for m in models}, Xt, r["y_test"],
                                    {m: r["test"][m] for m in models})
                tab["task"], tab["design"] = t, DESIGN_LABEL[tag]
                dis.append(tab)
                info.append({**inf, "task": t, "design": DESIGN_LABEL[tag]})
                for m in models:
                    cm.append({"task": t, "design": DESIGN_LABEL[tag], "model": m,
                               "source_observed_auroc": A.all_metrics(r["y_dev"], r["oof"][m], t, which=("AUROC",))["AUROC"],
                               "source_expected_auroc": SH.casemix_auroc(r["oof"][m]),
                               "target_observed_auroc": A.all_metrics(r["y_test"], r["test"][m], t, which=("AUROC",))["AUROC"],
                               "target_expected_auroc": SH.casemix_auroc(r["test"][m])})
                if tag == "primary":
                    m = res[t]["primary"]
                    iw = SH.iw_sensitivity(Xs, r["y_dev"], r["oof"][m], Xt, r["y_test"], r["test"][m])
                    iw["task"], iw["model"] = t, m
                    iws.append(iw)
            log(f"[shift {t}] DISDE done ({time.time() - t0:.0f}s)")
        pd.concat(dis).to_csv(T("shift_disde.csv"), index=False)
        pd.DataFrame(info).to_csv(T("shift_disde_info.csv"), index=False)
        pd.DataFrame(cm).to_csv(T("shift_casemix.csv"), index=False)
        pd.concat(iws).to_csv(T("shift_iw_sensitivity.csv"), index=False)
        log(f"--- shift done ({time.time() - t0:.0f}s)")
    if run("iwrecal"):
        rows = []
        for t in TASKS:
            d = task_frame(a, t)
            Xn = designs_for(d, t)["num"][0]
            for tag, r in (("primary", res[t]), ("c1to3", sec[("c1to3", t)]), ("c2to3", sec[("c2to3", t)]),
                           ("chain12", sec[("chain12", t)])):
                models = list(dict.fromkeys([m for m in ("LR", "RidgeOH", "XGB", res[t]["primary"]) if m in r["oof"]]))
                tab = U.iw_recal_eval(Xn.loc[r["dev_index"]], r["y_dev"], {m: r["oof"][m] for m in models},
                                      Xn.loc[r["test_index"]], r["y_test"], {m: r["test"][m] for m in models})
                tab["task"], tab["design"] = t, DESIGN_LABEL[tag]
                rows.append(tab)
            log(f"[iw-recal {t}] done ({time.time() - t0:.0f}s)")
        pd.concat(rows).to_csv(T("update_iw_recalibration.csv"), index=False)
    if run("lagged"):
        longs, summ = [], []
        for t in ("A1", "A2", "P"):
            r1 = sec[("c1to3", t)]
            m = res[t]["primary"] if res[t]["primary"] in r1["estimators"] else "XGB"
            d = task_frame(a, t)
            D = designs_for(d, t)
            Xn, Xo = D["num"][0], D["oh"][0]
            i1 = d.index[d.cohort == 1]
            i2 = d.index[d.cohort == 2]
            i3 = d.index[d.cohort == 3]
            Xb = Xo if m in ONEHOT_MODELS else Xn
            lg = U.lagged_updating(m, r1["params"][m], Xb.loc[i1], d.loc[i1, "y"].values, Xb.loc[i2],
                                   d.loc[i2, "y"].values, Xb.loc[i3], d.loc[i3, "y"].values, Xo.loc[i2], Xo.loc[i3],
                                   r1["estimators"][m], R=R_UPDATE)
            lg["task"], lg["model"] = t, m
            longs.append(lg)
            s = U.summarise_updating(lg, "No update (C1 model)")
            s["task"], s["model"] = t, m
            summ.append(s)
            log(f"[lagged {t}] {m} done ({time.time() - t0:.0f}s)")
        pd.concat(longs).to_csv(T("update_lagged_reps.csv"), index=False)
        pd.concat(summ).to_csv(T("update_lagged_summary.csv"), index=False)
    if run("withinc3"):
        longs, summ = [], []
        for t in ("A1", "A2"):
            r = res[t]
            m = r["primary"]
            d = task_frame(a, t)
            D = designs_for(d, t)
            X = D["oh" if m in ONEHOT_MODELS else "num"][0]
            lg = U.within_c3_updating(m, r["params"][m], X.loc[r["dev_index"]], r["y_dev"], X.loc[r["test_index"]],
                                      r["y_test"], r["test"][m], D["oh"][0].loc[r["test_index"]], R=R_UPDATE,
                                      sizes=(100, 250, 500, 1000) if t == "A1" else (100, 250, 500))
            lg["task"], lg["model"] = t, m
            longs.append(lg)
            s = U.summarise_updating(lg, "No update")
            s["task"], s["model"] = t, m
            summ.append(s)
            log(f"[within-C3 {t}] done ({time.time() - t0:.0f}s)")
        pd.concat(longs).to_csv(T("update_withinC3_reps.csv"), index=False)
        pd.concat(summ).to_csv(T("update_withinC3_summary.csv"), index=False)
    if run("fairness"):
        from rtw_config import FAIR_THRESHOLD
        rows = []
        for t in TASKS:
            r = res[t]
            te = task_frame(a, t).loc[r["test_index"]]
            for m in dict.fromkeys([r["primary"], "RidgeOH"]):
                rows.append(fairness(te, r["y_test"], r["test"][m], t, m, FAIR_THRESHOLD[t], B=B_BOOT))
                rows.append(fairness(te, r["y_test"], r["test"][m], t, m, FAIR_THRESHOLD[t], B=B_BOOT,
                                     w=te["wt_w2"].values, strata=te["strata"].values, label="weighted"))
        pd.concat(rows).to_csv(T("fairness.csv"), index=False)
        log(f"--- fairness done ({time.time() - t0:.0f}s)")
    if run("attribution"):
        for t in ("A1", "A2", "P"):
            A.attribution_stability(res[t], a, out, log=log)
    if run("extras"):
        A.within_strata(res["P"], a, out)
        A.attrition(a, out)
        A.weights_design(a, out)                     # round 2: Kish ESS, single-unit strata
        A.disde_restricted(res, sec, a, out, log=log)  # round 2: restricted-set prevalence decomposition
        log(f"--- extras done ({time.time() - t0:.0f}s)")
    if run("sensitivity"):
        for t in ("A1", "A2", "P"):
            A.sensitivity(res[t], a, out, log=log)
            A.seed_variability(res[t], a, out)
        log(f"--- sensitivity done ({time.time() - t0:.0f}s)")
    if run("report"):
        import rtw_figures
        import rtw_tables
        rtw_figures.make_all(out)
        rtw_tables.make_all(out)
    log(f"=== v2 run complete in {time.time() - t0:.0f}s")
    logf.close()


if __name__ == "__main__":
    main()

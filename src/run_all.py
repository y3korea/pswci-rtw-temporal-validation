"""Run the complete pre-specified analysis pipeline.

Environment variables:
    DATA_ROOT       folder containing the PSWCI cohort folders (raw SPSS files)
    RTW_OUT_DIR     where aggregated tables/figures are written
    RTW_CACHE_DIR   cache for fitted models / row-level predictions (never share)
    RTW_USE_PARQUET 1 = read DATA_ROOT/00_derived_data/pswci_harmonized_long.parquet if present
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # silence sklearn/lightgbm feature-name warnings in workers
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtw_config import B_BOOT, B_SHAP, R_UPDATE, SEED, paths  # noqa: E402
from rtw_data import build_analytic, load_long  # noqa: E402
import rtw_analyses as A  # noqa: E402
from rtw_develop import develop, params_table  # noqa: E402


def main():
    np.random.seed(SEED)
    data_root, out, cache = paths()
    logf = open(os.path.join(out, "run_log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    log(f"=== run start; B_BOOT={B_BOOT} R_UPDATE={R_UPDATE} B_SHAP={B_SHAP}")
    long = load_long(data_root)
    a = build_analytic(long)
    log(f"analytic rows (wave-1 participants) = {len(a)}; wave-2 participants = {int(a.part2.sum())}")
    A.descriptives(a, out)
    A.predictor_table(a, out)

    res = {}
    for task in ("A", "B"):
        res[task] = develop(a, task, (1, 2), 3, reduced=False, tag="primary", cache_dir=cache, log=log)
    pd.concat([params_table(res[t]) for t in res]).to_csv(os.path.join(out, "tables", "tuning_primary.csv"),
                                                          index=False)
    log(f"--- development done ({time.time() - t0:.0f}s)")
    for task in ("A", "B"):
        A.evaluate(res[task], out, B_BOOT, log=log)
    log(f"--- evaluation done ({time.time() - t0:.0f}s)")
    A.feature_shift(a, out)
    A.adversarial(a, out, B_BOOT)
    A.transitions(a, out)
    for task in ("A", "B"):
        A.importance_weighting(res[task], a, out, B_BOOT, log=log)
    log(f"--- shift done ({time.time() - t0:.0f}s)")
    for task in ("A", "B"):
        A.updating(res[task], a, out, R_UPDATE, log=log)
    log(f"--- updating done ({time.time() - t0:.0f}s)")
    prim_a = res["A"]["primary"]
    A.fairness(res["A"], a, out, [prim_a] + (["LR"] if prim_a != "LR" else []), 0.30, B_BOOT)
    A.fairness(res["A"], a, out, [prim_a] + (["LR"] if prim_a != "LR" else []), 0.50, B_BOOT)
    A.fairness(res["B"], a, out, [res["B"]["primary"]], 0.60, B_BOOT)
    log(f"--- fairness done ({time.time() - t0:.0f}s)")
    A.shap_stability(res["A"], a, out, B_SHAP, log=log)
    A.shap_stability(res["B"], a, out, max(20, B_SHAP // 5), log=log)
    log(f"--- SHAP done ({time.time() - t0:.0f}s)")
    for task in ("A", "B"):
        A.sensitivity(res[task], a, out, B_BOOT, log=log)
    log(f"--- sensitivity done ({time.time() - t0:.0f}s)")
    A.secondary_designs(a, out, B_BOOT, cache, log=log)
    log(f"--- secondary designs done ({time.time() - t0:.0f}s)")
    import rtw_figures
    import rtw_tables
    rtw_figures.make_all(out)
    rtw_tables.make_all(out)
    log(f"=== run complete in {time.time() - t0:.0f}s")
    logf.close()


if __name__ == "__main__":
    main()

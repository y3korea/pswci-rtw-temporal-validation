"""
pswci_harmonize.py
==================
Harmonize the three cohorts of the Korean Panel Study of Workers' Compensation
Insurance (PSWCI; 산재보험패널조사) into one long-format analysis file.

    Cohort 1: 2013-2017 (5 waves), workers whose compensated medical care ended in 2012
    Cohort 2: 2018-2022 (5 waves), care ended in 2017
    Cohort 3: 2023-2024 (2 waves), care ended in 2022

Usage (local or Google Colab):
    from pswci_harmonize import build_harmonized
    df = build_harmonized(DATA_ROOT)            # DATA_ROOT = folder that contains the cohort folders
    df.to_parquet("pswci_harmonized_long.parquet")

Coding traps handled here (each one was verified against the SPSS value labels):
  * Disability grade: cohort 1 codes "no disability" as 0, cohorts 2-3 as 15.
  * Self-rated health: cohorts 1-2 use 4 levels (1 = very poor ... 4 = very good);
    cohort 3 uses 5 levels in the opposite direction (1 = very healthy ... 5 = not at all).
  * Difficulty in daily activities: cohorts 1-2 code 1 = very difficult ... 5 = not at all;
    cohort 3 codes 1 = not at all ... 5 = very difficult.
  * Recovery after injury: cohort 1 has 3 levels (1 = fully recovered), cohorts 2-3 have
    5 levels (5 = fully recovered).
  * Age: cohort 1 has 5 groups (<=20s, 30s, ...), cohorts 2-3 have 4 groups (<=30s, ...).
  * Firm size: cohort 1 has 7 classes, cohorts 2-3 have 8 classes.
  * Rehabilitation-service flags: 1/2 in cohorts 1 and 3, 1/0 in cohort 2.
  * Smoking/drinking: G017/G018 in cohort 2 but G018/G019 in cohort 3 (G016/G017 = sleep).
  * Cohort 1 'accident' has no commuting category (commuting injuries cannot be identified
    and are coded 0); cohorts 2-3 code 3 = commuting (C2: pre-2018 employer-controlled
    commuting only).
  * "Long-lasting disability" block: item 1 = sensory impairment, item 2 = physical-activity
    limitation (walking/stairs/lifting). Use `phys_limitation` for physical limitation.
  * Cohort 1 wave-1 work ability refers to ~90 days after care closure (not the interview date).
Raw PSWCI data are public-use data (free download for anyone from
https://www.pswci.or.kr/Data/Download.aspx after accepting the conditions of use); because
the conditions prohibit transfer to third parties they are not redistributed with this code,
and this script only reads files the user downloaded.
"""
from __future__ import annotations

import os
import unicodedata

import numpy as np
import pandas as pd
import pyreadstat

# --------------------------------------------------------------------------------------
# File discovery (Google Drive on macOS stores Korean names in NFD; normalise before match)
# --------------------------------------------------------------------------------------

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def find_file(root: str, must_contain: list[str], ext: str = ".sav") -> str:
    hits = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            q = _nfc(p)
            if q.lower().endswith(ext) and all(_nfc(s) in q for s in must_contain):
                hits.append(p)
    if not hits:
        raise FileNotFoundError(f"No {ext} file containing {must_contain} under {root}")
    return sorted(hits, key=len)[0]


def cohort_paths(root: str) -> dict[int, str]:
    return {
        1: find_file(root, ["1차 코호트", "LONG_DATA"]),
        2: find_file(root, ["2차 코호트", "pswci2_05_long"]),
        3: find_file(root, ["3차 코호트", "(4. 통합 데이터_long)"], ext=".sav"),
    }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _valid(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Keep values inside [lo, hi]; refusal / don't-know codes (8, 9, 98, 99, ...) -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= lo) & (s <= hi))


def _yes(s: pd.Series, yes=1, no=(2,)) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    out[s == yes] = 1.0
    out[s.isin(no)] = 0.0
    return out


def _read(path: str, cols: list[str]) -> pd.DataFrame:
    _, meta = pyreadstat.read_sav(path, metadataonly=True)
    have = [c for c in cols if c in meta.column_names]
    missing = sorted(set(cols) - set(have))
    if missing:
        print(f"[harmonize] note: {os.path.basename(path)} lacks {missing}")
    df, _ = pyreadstat.read_sav(path, usecols=have)
    for c in missing:
        df[c] = np.nan
    return df


# Column maps: harmonised name -> cohort-specific source column
MAP = {
    1: dict(p="p", wave="wave", sex="gender", age_raw="age", edu="edu", tenure="workperiod",
            accident="accident", care_dur="con", disa15="disa15",
            svc_any="service", svc_job="jobservice", svc_med="medservice", svc_soc="psyservice",
            emp3="emp03", emp6="emp06", wt_long="wt", wt_cross="wt",
            ex_days="Gb003003", ex_h="Gb003004", ex_m="Gb003005",
            work_ability="Bb007002", work_ability_postcare="Bb007001",
            srh="Gb001001", chronic="Gb001003", limitation="Gb003001", phys_lim="Gb003002", adl_help="Gb004001",
            adl_mental="Gb005001", adl_indoor="Gb005002", adl_outdoor="Gb005003", adl_work="Gb005004",
            life_sat="Gb008007", smoke="Gb010001", drink="Gb010007",
            recovery="Ga001002", pain_freq="Ga001003", pain_interf="Ga001004", preinj_health="Ga002001",
            firm_size="C002004", union="C002019", job_status="C001007", industry="C001003",
            marital="A003008", hh_size="I001001"),
    2: dict(p="p", wave="wave", sex="gender", age_raw="age4", edu="edu", tenure="workperiod14",
            accident="accident", injury_type="injurytype", injury_part="injurypart",
            care_dur="con16", disa15="disa15",
            svc_any="service", svc_job="jobservice", svc_soc="socservice",
            emp3="emp3", emp6="emp6", wt_long="wt1", wt_cross="wt2",
            ex_days="G015001", ex_h="G015002", ex_m="G015003",
            work_ability="B026001", work_ability_postcare="B025001",
            srh="G005001", chronic="G006001", limitation="G009001", phys_lim="G009002", adl_help="G010001",
            adl_mental="G012001", adl_indoor="G012002", adl_outdoor="G012003", adl_work="G012004",
            life_sat="G022007", smoke="G017001", drink="G018001",
            recovery="G001001", pain_freq="G002001", pain_interf="G003001", preinj_health="G004001",
            firm_size="C024001", union="C011001", job_status="C005001", industry="C002002",
            marital="A002002", hh_size="I001001"),
    3: dict(p="p", wave="wave", sex="gender", age_raw="age4", edu="edu", tenure="workperiod14",
            accident="accident", injury_type="injurytype", injury_part="injurypart",
            care_dur="con6", disa15="disa15",
            svc_any="service", svc_job="jobservice", svc_med="medservice", svc_soc="socservice",
            svc_owner="ownerservice",
            emp3="emp3", emp6="emp6", wt_long="wt1", wt_cross="wt2", strata="strata",
            ex_days="G016001", ex_h="G016002", ex_m="G016003",
            work_ability="B026001", work_ability_postcare="B025001",
            srh="G005001", chronic="G006001", limitation="G010001", phys_lim="G010002", adl_help="G012001",
            adl_mental="G014001", adl_indoor="G014002", adl_outdoor="G014003", adl_work="G014004",
            life_sat="G026007", smoke="G018001", drink="G019001",
            recovery="G001001", pain_freq="G002001", pain_interf="G003001", preinj_health="G004001",
            preinj_chronic="G004002",
            firm_size="C030001", union="C014001", job_status="C005001", industry="C002003",
            marital="A002002", hh_size="I001001",
            **{f"phq{i}": f"G0290{i:02d}" for i in range(1, 10)}),
}

# Variables measured (by design) only at wave 1 -> carried as baseline attributes
WAVE1_ONLY = ["recovery", "pain_freq", "pain_interf", "preinj_health", "preinj_chronic",
              "work_ability_postcare", "firm_size", "union", "job_status", "industry"]
# Administrative / sampling attributes fixed at baseline
FIXED = ["sex", "age_raw", "edu", "tenure", "accident", "injury_type", "injury_part",
         "care_dur", "disa15", "svc_any", "svc_job", "svc_med", "svc_soc", "svc_owner", "strata"]


def _load_cohort(root: str, cohort: int, path: str) -> pd.DataFrame:
    m = MAP[cohort]
    raw = _read(path, ["pid"] + list(m.values()))
    df = pd.DataFrame({"pid": raw["pid"].astype("int64")})
    for k, src in m.items():
        df[k] = raw[src].values
    df["cohort"] = cohort
    for k in set().union(*[set(v) for v in MAP.values()]) - set(m):
        df[k] = np.nan  # item not fielded in this cohort
    # broadcast wave-1 values of baseline-only items to every wave of the same person
    w1 = df[df["wave"] == 1].set_index("pid")
    for k in WAVE1_ONLY + FIXED:
        if k in df.columns:
            df[k] = df["pid"].map(w1[k])
    return df


def _harmonize(df: pd.DataFrame) -> pd.DataFrame:
    c = df["cohort"]
    out = pd.DataFrame({"cohort": c, "pid": df["pid"], "wave": df["wave"].astype(int)})
    out["uid"] = out["cohort"] * 100000 + out["pid"]
    out["year"] = np.select([c == 1, c == 2, c == 3], [2012, 2017, 2022]) + out["wave"]
    out["participated"] = (pd.to_numeric(df["p"], errors="coerce") == 1).astype(int)
    out["wt_long"] = pd.to_numeric(df["wt_long"], errors="coerce")
    out["wt_cross"] = pd.to_numeric(df["wt_cross"], errors="coerce")
    out["strata"] = df["strata"]

    # ---- demographics -----------------------------------------------------------------
    out["female"] = (pd.to_numeric(df["sex"], errors="coerce") == 2).astype(float)
    age = pd.to_numeric(df["age_raw"], errors="coerce")
    out["age_grp"] = np.where(c == 1, age.map({1: 1, 2: 1, 3: 2, 4: 3, 5: 4}), age)  # 1<=39 2=40s 3=50s 4>=60
    out["edu"] = _valid(df["edu"], 1, 5)                                               # 1 none ... 5 college+
    out["married"] = (pd.to_numeric(df["marital"], errors="coerce") == 2).astype(float).where(df["marital"].notna())
    out["hh_size"] = _valid(df["hh_size"], 1, 20)

    # ---- injury & compensation (baseline, administrative) ------------------------------
    acc = pd.to_numeric(df["accident"], errors="coerce")
    out["accident_type"] = acc                                    # 1 accident, 2 disease, 3 commuting (c2-3)
    out["disease"] = (acc == 2).astype(float)
    out["commuting"] = (acc == 3).astype(float)
    out["injury_type"] = pd.to_numeric(df["injury_type"], errors="coerce")
    part = pd.to_numeric(df["injury_part"], errors="coerce")
    out["injury_part"] = part
    region = pd.Series(np.nan, index=df.index, dtype=object)
    region[part.isin([5, 9])] = "spine"          # neck, low back
    region[part.isin([6, 7])] = "upper_limb"     # arm, hand/finger
    region[part.isin([10, 11, 12])] = "lower_limb"  # hip, leg, foot/toe
    region[part.isin([8, 19])] = "trunk"         # chest/back, abdomen
    region[part.isin([1, 2, 3, 4])] = "head_face"
    region[part == 13] = "multiple"
    region[part.isin([14, 15, 16, 17, 18, 20, 21])] = "systemic_other"
    out["injury_region"] = region
    out["care_dur"] = _valid(df["care_dur"], 1, 6)               # 1 <=3m ... 6 >2y
    out["tenure"] = _valid(df["tenure"], 1, 14)                   # 1 <1m ... 14 >=20y
    d15 = pd.to_numeric(df["disa15"], errors="coerce")
    grade = np.where(c == 1, d15.where(d15 >= 1), d15.where(d15 <= 14))
    out["disab_grade"] = grade                                    # 1 (most severe) ... 14; NaN = none
    out["disab_any"] = pd.Series(grade, index=df.index).notna().astype(float)
    g = pd.Series(grade, index=df.index)
    out["disab_cat"] = pd.cut(g, [0, 3, 7, 9, 12, 14], labels=[1, 2, 3, 4, 5]).astype(float).fillna(6)
    # 1: grade 1-3, 2: 4-7, 3: 8-9, 4: 10-12, 5: 13-14, 6: none

    for k in ["svc_any", "svc_job", "svc_med", "svc_soc", "svc_owner"]:
        if k not in df:
            out[k] = np.nan
            continue
        s = pd.to_numeric(df[k], errors="coerce")
        out[k] = np.where(c == 2, s.map({1: 1.0, 0: 0.0}), s.map({1: 1.0, 2: 0.0}))
    # cohort 2 has no medical-rehabilitation flag; cohort 1 "psyservice" = social/psychological

    # ---- job at time of injury (asked at wave 1) ----------------------------------------
    fs = pd.to_numeric(df["firm_size"], errors="coerce")
    out["firm_size"] = np.where(c == 1, fs.map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6}),
                                fs.map({1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6}))
    # 1 <5, 2 5-9, 3 10-29, 4 30-99, 5 100-299, 6 300+
    out["union"] = _yes(df["union"], 1, (2, 3))
    js = pd.to_numeric(df["job_status"], errors="coerce")
    out["job_status"] = js.map({1: 1, 2: 2, 3: 3, 4: 4, 5: 4})  # 1 regular 2 temporary 3 daily 4 self-employed
    ind = pd.to_numeric(df["industry"], errors="coerce")
    out["industry"] = ind.map(lambda x: {3: "manufacturing", 6: "construction", 8: "transport"}.get(x, "other")
                              if pd.notna(x) else np.nan)

    # ---- economic activity (all waves) --------------------------------------------------
    out["emp3"] = _valid(df["emp3"], 1, 3)       # 1 employed, 2 unemployed, 3 not in labour force
    out["emp6"] = _valid(df["emp6"], 1, 6)       # 1 original job, 2 re-employed, 3 self-employed, ...
    out["employed"] = (out["emp3"] == 1).astype(float).where(out["emp3"].notna())
    out["rtw_original"] = (out["emp6"] == 1).astype(float).where(out["emp6"].notna())

    # ---- exercise (all waves) -----------------------------------------------------------
    days = _valid(df["ex_days"], 0, 7)
    h = _valid(df["ex_h"], 0, 16)
    mnt = _valid(df["ex_m"], 0, 59)
    per_day = (h.fillna(0) * 60 + mnt.fillna(0)).where(h.notna() | mnt.notna())
    per_day = per_day.where(days > 0, 0.0).where(days.notna())
    out["ex_days"] = days
    out["ex_min_day"] = per_day
    out["ex_min_week"] = days * per_day
    out["inactive"] = (days == 0).astype(float).where(days.notna())
    out["regular_ex"] = ((days >= 3) & (per_day >= 30)).astype(float).where(days.notna())
    out["ex_150"] = (out["ex_min_week"] >= 150).astype(float).where(out["ex_min_week"].notna())

    # ---- function, health (all waves) ---------------------------------------------------
    out["work_ability"] = _valid(df["work_ability"], 0, 10)
    # Cohort 1, wave 1 asked work ability "about 90 days after care closure"; from wave 2 (and in
    # cohorts 2-3 at every wave) it refers to the interview date.
    out["work_ability_ref_interview"] = (~((c == 1) & (out["wave"] == 1))).astype(int)
    out["work_ability_postcare"] = _valid(df["work_ability_postcare"], 0, 10)
    srh = pd.to_numeric(df["srh"], errors="coerce")
    out["poor_health"] = np.where(c == 3, srh.map({1: 0, 2: 0, 3: 0, 4: 1, 5: 1}),
                                  srh.map({1: 1, 2: 1, 3: 0, 4: 0}))
    out["chronic"] = _yes(df["chronic"])
    # Item 1 of the "long-lasting disability or physical restriction" block = SENSORY impairment
    # (vision/hearing/speech); item 2 = substantial limitation in walking, climbing stairs,
    # lifting/carrying. `limitation` is kept for backward compatibility and equals item 1.
    out["limitation"] = _yes(df["limitation"])          # DEPRECATED name: sensory impairment
    out["sensory_impair"] = _yes(df["limitation"])
    out["phys_limitation"] = _yes(df["phys_lim"])
    out["adl_help"] = _valid(df["adl_help"], 1, 4)                    # higher = more help needed
    out["adl_help_need"] = (out["adl_help"] >= 3).astype(float).where(out["adl_help"].notna())
    for k in ["adl_mental", "adl_indoor", "adl_outdoor", "adl_work"]:
        v = _valid(df[k], 1, 5)
        out[k] = np.where(c == 3, v, 6 - v)                             # harmonised: 1 none ... 5 very difficult
        out[k + "_diff"] = (pd.Series(out[k], index=df.index) >= 4).astype(float).where(v.notna())
    ls = _valid(df["life_sat"], 1, 5)
    out["life_sat"] = 6 - ls                                            # higher = more satisfied
    out["smoker"] = _yes(df["smoke"])
    out["drinker"] = _yes(df["drink"], 1, (2, 3))  # cohort 1 has an unlabelled code 3 (19% at wave 1); questionnaire asks about quitting -> treated as not a current drinker

    # ---- wave-1-only items (broadcast as baseline attributes) ---------------------------
    rec = pd.to_numeric(df["recovery"], errors="coerce")
    out["fully_recovered"] = np.where(c == 1, rec.map({1: 1.0, 2: 0.0, 3: 0.0}),
                                      rec.map({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0}))
    out["pain_freq"] = _valid(df["pain_freq"], 1, 5)                   # 1 never ... 5 always
    out["pain_interf"] = _valid(df["pain_interf"], 1, 4)               # 1 not at all ... 4 always
    out["pain_interf_high"] = (out["pain_interf"] >= 3).astype(float).where(out["pain_interf"].notna())
    out["preinj_poor_health"] = (_valid(df["preinj_health"], 1, 5) >= 4).astype(float).where(
        _valid(df["preinj_health"], 1, 5).notna())
    out["preinj_chronic"] = _yes(df["preinj_chronic"])

    # ---- cohort-3-only extras -----------------------------------------------------------
    if all(f"phq{i}" in df for i in range(1, 10)):
        items = pd.concat([_valid(df[f"phq{i}"], 1, 4) - 1 for i in range(1, 10)], axis=1)
        out["phq9"] = items.sum(axis=1, min_count=9)
    else:
        out["phq9"] = np.nan

    # non-participants carry no wave-specific measurements
    tv = ["emp3", "emp6", "employed", "rtw_original", "ex_days", "ex_min_day", "ex_min_week",
          "inactive", "regular_ex", "ex_150", "work_ability", "poor_health", "chronic", "limitation",
          "sensory_impair", "phys_limitation", "adl_help", "adl_help_need", "adl_mental", "adl_indoor", "adl_outdoor", "adl_work",
          "adl_mental_diff", "adl_indoor_diff", "adl_outdoor_diff", "adl_work_diff", "life_sat",
          "smoker", "drinker", "married", "hh_size", "phq9"]
    out.loc[out["participated"] == 0, tv] = np.nan
    return out


def build_harmonized(root: str) -> pd.DataFrame:
    paths = cohort_paths(root)
    frames = [_harmonize(_load_cohort(root, k, p)) for k, p in paths.items()]
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if df[col].dtype == object and col not in ("injury_region", "industry", "strata"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["cohort", "pid", "wave"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "pswci_harmonized_long.parquet"
    d = build_harmonized(root)
    d.to_parquet(out, index=False)
    print(d.shape, "->", out)

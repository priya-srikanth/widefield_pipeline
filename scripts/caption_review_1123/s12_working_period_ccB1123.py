"""S12 (recovery mechanics, six-position cohort) on the camera WORKING PERIOD (CC-B overnight 11-23, item 3f).

Priya 11-23: every six-position trial gate = the camera working period (the camera-tagged attempts table on the
3d_mouth jaw rule, `canonical` = working period), not the near-spout DAQ reference gate. The live S12 tables
(cam3on, WLC_LAUNCH=2v, 4,000 draws; lp_stage/phaseB_20261122/cam3on_20261123/driver.py) used:
  within_lick_correction  its own camera working period (`trial_classes.working_trial_ids`, in-process classes on the
                          pre-11-23-05:00 jaw rule), applied when the per-lick pickles are READ (`load_licks`)
  learned_compensation    `trial_attempts.csv` `canonical` (gate "reference" on every row) for T1 trials and the
                          previous trial of every T3 ITI lick (`prev_canonical`), plus the WLC licks_scored above
  lick_correction step_b  no trial gate (every post-cue lick pair of the curated sessions)
Nothing upstream depends on the gate: the WLC per-lick pickles (keyed by trial_id; inputs = 3-D frames + lick table)
and the LC timing pickles (DAQ timing + lick table) are reused as they are. Only the trial filter is swapped, then
the summaries and the stats are recomputed with the production code (patched at the one filter function each).

  wlc   --mask prod|wp --out DIR   within_lick_correction summarise; mask prod = the trial set of the live
                                   licks_scored (reproduction), wp = new attempts `canonical`
  lc    --gate prod|wp --wlc DIR --out DIR   learned_compensation run + summarise (timing pickles copied, not moved)
  recmech --wlc DIR --lc DIR --out DIR       c2_recovery_mechanics_stats.py --n-boot 4000 (live family / lickcorr)
  compare A B                      cell-by-cell diff of two output folders' csv tables (max |diff|)
New folders only; refuses to write into an existing non-empty output folder (except `lc` into the folder it created).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M = "M:/MICROSCOPE/Priya"
R = f"{M}/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60"
E = Path(f"{R}/imaging_encoding")
S = Path(f"{M}/Widefield/manuscript_stats/c2_4cam_sync60_20261122")
PROD_ATT = E / "qc" / "trial_attempts.csv"
NEW_ATT = (E / "qc" / "jaw3dmouth_ccB1123" / "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
           / "trial_attempts.csv")
PROD_WLC = E / "within_lick_correction_fourcam_launch2v_cam3on"
PICKLES = E / "within_lick_correction_fourcam_launchdefault_cam3on" / "licks_fused_52c775bb"
PROD_LC = E / "learned_compensation_fourcam_launch2v_cam3on"
LICKCORR = E / "iti_licks_cam3on" / "lick_correction"
FAMILY = S / "scalar_family_final_boot4000_cam3on" / "c2_scalar_family_aligned.csv"

ENCENV = {"OROFACIAL_GATE": "attempted", "COMBINED_ENCODING_LICKS": "fourcam", "COMBINED_ENCODING_ONLY": "full",
          "COMBINED_ENCODING_SUFFIX": "_gate-attempted", "COMBINED_ENCODING_TONGUE": "3d", "FROZEN_FR": "1",
          "COMBINED_ENCODING_FR": "master_pre_v1", "COMBINED_ENCODING_FR_GROUPS": "master_v1:A",
          "WFIELD_INPUT_CACHE": "C:/Users/SabatiniLab/wfield_cache", "WIDEFIELD_MAX_JOBS": "3",
          "WIDEFIELD_FROZEN_MODEL_DIR": f"{R}/frozen_models", "FROZEN_LICKSPOUT": "1"}
STRIP = ("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_", "FROZEN_", "WLC_", "LC_", "GLMHMM_", "REVUP_")


def _env(extra: dict) -> None:
    """The live driver's environment (ENCENV, every pipeline switch stripped first), set BEFORE the imports."""
    for k in [k for k in os.environ if k.startswith(STRIP)]:
        del os.environ[k]
    os.environ.update({**ENCENV, **extra})


def _b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def _new_out(d: Path) -> None:
    if d.exists() and any(d.iterdir()):
        sys.exit(f"refusing: {d} exists and is not empty")
    d.mkdir(parents=True, exist_ok=True)


def wp_gate() -> pd.DataFrame:
    """[label, trial_id, canonical, has_cam] from the new camera-tagged attempts table (one row per trial)."""
    A = pd.read_csv(NEW_ATT, low_memory=False, usecols=lambda c: c in ("label", "trial_id", "canonical", "gate",
                                                                      "cls", "cls_raw"))
    assert A.gate.astype(str).str.startswith("camera:").all(), A.gate.unique()[:5]
    assert not A.duplicated(["label", "trial_id"]).any()
    A["canonical"] = _b(A.canonical)
    raw = A.cls_raw if "cls_raw" in A else A.cls
    A["has_cam"] = A.label.map((raw != "unknown").groupby(A.label).any())
    return A[["label", "trial_id", "canonical", "has_cam"]]


# ------------------------------------------------------------------------------------------------ within_lick_correction
def cmd_wlc(mask: str, out: Path) -> None:
    _new_out(out)
    shutil.copy2(PROD_WLC / "csm_threshold.json", out / "csm_threshold.json")   # calibration as the live run (fixed)
    _env({"WLC_OUT_DIR": str(out), "WLC_LAUNCH": "2v", "WLC_LICKS_DIR": str(PICKLES)})
    import within_lick_correction as W
    assert Path(W.OUT) == out and Path(W.LICKS) == PICKLES, (W.OUT, W.LICKS)
    if mask == "prod":
        L = pd.read_csv(PROD_WLC / "licks_scored.csv.gz", usecols=["sid", "trial_id"])
        keep = {s: set(g.trial_id.astype(int)) for s, g in L.groupby("sid")}

        def m(sid, trial_id):
            k = keep.get(sid, set())
            return np.array([int(t) in k for t in np.asarray(trial_id)], bool)
    else:
        G = wp_gate()
        cam = G.groupby("label").has_cam.first().to_dict()
        keep = {lab: set(g.trial_id[g.canonical].astype(int)) for lab, g in G.groupby("label")}

        def m(sid, trial_id):
            lab = W._label(sid)
            if not cam.get(lab, False):
                return None                              # no camera classes: left out, as the live rule
            k = keep[lab]
            return np.array([int(t) in k for t in np.asarray(trial_id)], bool)
    W.working_mask = m
    W.cmd_summarise()
    (out / "GATE.txt").write_text(f"mask={mask}\nattempts={NEW_ATT if mask == 'wp' else 'live licks_scored trial set'}\n"
                                  f"pickles={PICKLES}\n")


# ------------------------------------------------------------------------------------------------ learned_compensation
def cmd_lc(gate: str, wlc: Path, out: Path) -> None:
    _new_out(out)
    tim = out / "timing"
    tim.mkdir()
    for f in (PROD_LC / "timing").glob("*.pkl"):                 # copy (not move): DAQ timing only, no gate
        shutil.copy2(f, tim / f.name)
    A = pd.read_csv(PROD_ATT, low_memory=False)
    if gate == "wp":
        G = wp_gate()
        A = A.merge(G[["label", "trial_id", "canonical"]], on=["label", "trial_id"], how="left", suffixes=("_ref", ""),
                    validate="one_to_one")
        miss = A.canonical.isna()
        print(f"attempts rows {len(A)}; without a row in the camera table: {int(miss.sum())} (-> not canonical)")
        A["canonical"] = A.canonical.fillna(False).astype(bool)
        A["gate"] = "camera working period (jaw3dmouth_ccB1123 attempts canonical); cls / hit from the live table"
        print("canonical ref -> wp:", pd.crosstab(_b(A.canonical_ref), A.canonical).to_dict())
    att = out / "trial_attempts_gate_used.csv.gz"
    A.to_csv(att, index=False)
    _env({"WLC_OUT_DIR": str(wlc), "LC_OUT_DIR": str(out)})
    from wfield_local import lp_licks
    lp_licks.attempts_path = lambda rv=None: att
    sys.argv = [sys.argv[0], "run", "--jobs", "3"]
    import learned_compensation as LCm
    assert Path(LCm.OUT) == out and Path(LCm.WLC_DIR) == wlc, (LCm.OUT, LCm.WLC_DIR)
    LCm.cmd_run(3)
    LCm.cmd_summarise()


# ------------------------------------------------------------------------------------------------ stats
def cmd_recmech(wlc: Path, lc: Path, out: Path) -> None:
    _new_out(out)
    _env({"WLC_OUT_DIR": str(wlc), "LC_OUT_DIR": str(lc)})
    import c2_recovery_mechanics_stats as RM
    RM.main(["--enc", str(E), "--wlc", str(wlc), "--lc", str(lc), "--lickcorr", str(LICKCORR), "--family", str(FAMILY),
             "--out-dir", str(out), "--ms-data", str(out / "ms_data"), "--n-boot", "4000"])


def cmd_compare(a: Path, b: Path) -> None:
    for fa in sorted(a.glob("*.csv")):
        fb = b / fa.name
        if not fb.exists():
            continue
        x, y = pd.read_csv(fa, low_memory=False), pd.read_csv(fb, low_memory=False)
        if x.shape != y.shape or list(x.columns) != list(y.columns):
            print(f"{fa.name}: shape / columns differ {x.shape} vs {y.shape}")
            continue
        num = x.select_dtypes("number").columns
        d = (x[num] - y[num]).abs().max().max() if len(num) else 0.0
        obj = [c for c in x.columns if c not in num]
        ne = int((x[obj].astype(str) != y[obj].astype(str)).any(axis=1).sum()) if obj else 0
        print(f"{fa.name}: rows {len(x)}, max |num diff| {d:.3g}, text rows differing {ne}")


def main() -> None:
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("wlc")
    p.add_argument("--mask", choices=["prod", "wp"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("lc")
    p.add_argument("--gate", choices=["prod", "wp"], required=True)
    p.add_argument("--wlc", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("recmech")
    p.add_argument("--wlc", type=Path, required=True)
    p.add_argument("--lc", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("compare")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    a = ap.parse_args()
    if a.cmd == "wlc":
        cmd_wlc(a.mask, a.out)
    elif a.cmd == "lc":
        cmd_lc(a.gate, a.wlc, a.out)
    elif a.cmd == "recmech":
        cmd_recmech(a.wlc, a.lc, a.out)
    else:
        cmd_compare(a.a, a.b)


if __name__ == "__main__":
    main()

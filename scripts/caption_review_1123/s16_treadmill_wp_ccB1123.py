"""S16 six-position treadmill (CC-B overnight item 3d, 11-23): does the treadmill trial gate change under the adopted
jaw rule (``WIDEFIELD_JAW_CALL=3d_mouth`` + the cap jaw table), and what do the readouts give on the camera WORKING
PERIOD of `trial_attempts` (``canonical``) instead of the treadmill's own lick span?

The treadmill scripts (`c2_treadmill_cue_aligned.py`, `c2_treadmill_extras.py`, `c2_treadmill_cue_aligned_first95.py`)
gate trials on the collector's ``lick_period`` column (`behavior_states.collect_session`) =
`trial_classes.camera_engaged_period` = first to LAST camera lick (contact | incomplete) of the classified train, one
start and one end boundary; a sustained mid-session stop stays inside it. Neither the jaw rule nor jaw-only enters it.

    period  <out.csv>   per trial (collector label, trial_id): live lick span + live classified class under the
                        CURRENT env (run once with production env, once with the 3d_mouth env)
    compare --old A.csv --new B.csv --attempts <trial_attempts.csv> --out <dir> --suffix S
                        gate-change tally (collector vs live old vs live new; span and canonical working period) and,
                        for the canonical working period (new classes), the treadmill readouts + tests exactly as the
                        producing scripts (per_session, session_change_tests, n_boot 4,000, seed 0): primary
                        (fdr_scope all), secondary (BH within measure), primary first 95 min. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import behavior_states as B

PRIMARY = ["push_cue_minus_control", "walk_cue"]
SECONDARY = ["push_rate", "push_control_rate", "push_peak", "push_latency_s", "walk_control"]


def period(out: Path) -> None:
    from wfield_local import trial_classes as tc
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    C = B.load_collected()["trials"]
    rows = []
    for lab in sorted(C.label.unique()):
        T = tc.session_trials(lab)
        if T is None or not len(T):
            continue
        T = T.sort_values("trial_id").reset_index(drop=True)
        cue = T.cue_daq.to_numpy(float)
        rows.append(pd.DataFrame({"label": lab, "trial_id": T.trial_id.to_numpy(), "cue_daq": cue,
                                  "lick_period_live": np.asarray(tc.camera_engaged_period(lab, cue), bool),
                                  "cls_live": tc.cue_classes_for(lab, cue)}))
        print(lab, flush=True)
    D = pd.concat(rows, ignore_index=True)
    D["canonical_live"] = ~D.cls_live.isin(tc.OUT_OF_WORKING_PERIOD)
    D.to_csv(out, index=False)


def _tests(S, ms, scope):
    from wfield_local import orofacial_stats as st
    return st.session_change_tests(S, ms, fdr_scope=scope) if scope else st.session_change_tests(S, ms)


def _sessions(T, gate):
    S = B.per_session(T, gate)
    S["pos"] = "all"
    S["date"] = S.label.str.split("_").str[1]
    return S


def compare(a) -> None:
    outs = {k: a.out / f"{k}_{a.suffix}.csv" for k in (
        "treadmill_gate_change", "treadmill_session_readouts", "treadmill_session_readouts_first95",
        "treadmill_tests_primary", "treadmill_tests_secondary", "treadmill_tests_primary_first95")}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite {clash}")
    C = B.load_collected()["trials"]
    C = C[C.label.map(B.session_epoch).notna()].copy()
    C["epoch"] = C.label.map(B.session_epoch)
    k = ["label", "trial_id"]
    O = pd.read_csv(a.old).rename(columns=lambda c: c.replace("_live", "_old"))
    N = pd.read_csv(a.new).rename(columns=lambda c: c.replace("_live", "_new"))
    A = pd.read_csv(a.attempts, usecols=["label", "trial_id", "canonical", "cls"], low_memory=False)
    A["label"] = A.label.str.replace(r"^(PS\d+)_(\d{4})$", r"\1_2026\2", regex=True)
    A = A.rename(columns={"canonical": "canonical_att", "cls": "cls_att"})
    A["canonical_att"] = A.canonical_att.astype(str).str.lower().isin(["true", "1", "1.0"])
    M = C.merge(O.drop(columns="cue_daq"), on=k, how="left").merge(N, on=k, how="left").merge(A, on=k, how="left")
    miss = M[["lick_period_old", "lick_period_new", "canonical_att"]].isna().sum().to_dict()
    print("collector trials", len(M), "unmatched", miss)
    for c in ("lick_period_old", "lick_period_new", "canonical_old", "canonical_new", "canonical_att"):
        M[c] = M[c].astype("boolean").fillna(False).astype(bool)
    checks = {"collector lick_period vs live span, production classes": (M.lick_period != M.lick_period_old),
              "live span, production vs 3d_mouth classes": (M.lick_period_old != M.lick_period_new),
              "live canonical, production vs 3d_mouth classes": (M.canonical_old != M.canonical_new),
              "live canonical (3d_mouth) vs trial_attempts canonical": (M.canonical_new != M.canonical_att),
              "treadmill span vs canonical working period (3d_mouth)": (M.lick_period != M.canonical_new)}
    rows = []
    for name, d in checks.items():
        for ep, g in M.assign(d=d).groupby("epoch"):
            rows.append({"check": name, "epoch": ep, "n_trials": len(g), "n_changed": int(g.d.sum())})
        print(f"{name}: {int(d.sum())} of {len(M)}")
    G = pd.DataFrame(rows)
    span_only = M[M.lick_period & ~M.canonical_new]
    print("span but not canonical, by class:", span_only.cls_new.value_counts().to_dict())
    print("canonical but not span:", int((~M.lick_period & M.canonical_new).sum()))
    G.to_csv(outs["treadmill_gate_change"], index=False)
    M["wp_canonical"] = M.canonical_new
    Sall, S95 = _sessions(M, "wp_canonical"), _sessions(M[~(M.t_min > B.COMMON_SESSION_MIN)], "wp_canonical")
    Sall.to_csv(outs["treadmill_session_readouts"], index=False)
    S95.to_csv(outs["treadmill_session_readouts_first95"], index=False)
    _tests(Sall, PRIMARY, "all").to_csv(outs["treadmill_tests_primary"], index=False)
    _tests(Sall, SECONDARY, None).to_csv(outs["treadmill_tests_secondary"], index=False)
    _tests(S95, PRIMARY, "all").to_csv(outs["treadmill_tests_primary_first95"], index=False)
    for p in outs.values():
        print("->", p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("period")
    p.add_argument("out", type=Path)
    c = sub.add_parser("compare")
    for n in ("old", "new", "attempts", "out"):
        c.add_argument(f"--{n}", type=Path, required=True)
    c.add_argument("--suffix", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "period":
        period(a.out)
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        compare(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

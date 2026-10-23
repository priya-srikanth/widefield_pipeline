"""Backward / postural treadmill PUSHES as a behavioural readout, per session (Priya 10-23, option 3 of "let's do all 3":
pushes become 2-5x more frequent after the stroke in 3/4 animals -- does that track recovery or the far-contra
deficit?). Side-camera clips (10-23): negative speed at bout ends = a paw pushed forward; the post-cue dip = upper-limb
extension / posture change. Both are real movements, not belt backlash.

Per session, the signed treadmill speed (DAQ analog, the encoder's running regressor; negative = backward) on a
31.25 Hz grid over the trial span (first cue - 5 s .. last cue + 10 s):
  push_frac      fraction of samples with backward speed > PUSH_FLOOR (0.3 mm/s, frozen_encoder.PUSH_FLOOR)
  push_rate_min  push bouts per minute (runs of >= 2 samples beyond the floor)
  push_amp       mean backward speed over push samples, mm/s
  run_frac       fraction of samples running forward > 1.5 mm/s (the v7 running onset), for contrast
  cue_push_frac  fraction of trials with a push in the 1 s after the cue (the cue-locked postural dip)
  cue_dip        median over trials of (min speed 0-1 s after the cue - mean speed -0.5-0 s), mm/s (negative = dip)
  cue_push_frac_far_contra / cue_dip_far_contra   the same on far-contra trials only
-> <run>/imaging_encoding/qc/push_behaviour_sessions.csv; the LMMs: `python scripts/imaging_lmm.py --sources push`
(TIME and BEHAVIOUR models, per-animal table first).
Usage: python scripts/push_behaviour.py [--jobs N]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

FS = 31.25
PUSH_FLOOR = 0.3          # = frozen_encoder.PUSH_FLOOR
RUN_ON = 1.5              # = frozen_encoder.RUN_KNOT0_DEFAULT
FAR_CONTRA = ("far_contra", "far_R")      # lesion LEFT: contra = the mouse's right


def _bouts(m: np.ndarray, min_len: int = 2) -> int:
    d = np.diff(np.r_[0, m.astype(int), 0])
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int(((off - on) >= min_len).sum())


def _cue_stats(g: np.ndarray, s: np.ndarray, cues: np.ndarray) -> tuple[float, float]:
    post = np.arange(0, int(FS))
    pre = np.arange(-int(0.5 * FS), 0)
    i0 = np.searchsorted(g, cues)
    ok = (i0 + pre[0] >= 0) & (i0 + post[-1] < len(g))
    if not ok.any():
        return np.nan, np.nan
    i0 = i0[ok]
    sp, sb = s[i0[:, None] + post], s[i0[:, None] + pre]
    return float(np.mean((-sp > PUSH_FLOOR).any(1))), float(np.median(sp.min(1) - sb.mean(1)))


def job(label: str) -> dict:
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    B = ce.load_behaviour(an, date, PathResolver())
    tr, vr = (np.asarray(x, float) for x in B.analog["running"])
    T = B.trials
    cue = T.cue_s.to_numpy(float)
    ok = np.isfinite(cue)
    g = np.arange(cue[ok].min() - 5, cue[ok].max() + 10, 1 / FS)
    s = np.interp(g, tr, vr)
    push = -s > PUSH_FLOOR
    pos = T.pos.astype(str).to_numpy()
    fc = ok & np.isin(pos, FAR_CONTRA)
    cp, cd = _cue_stats(g, s, cue[ok])
    cpf, cdf = _cue_stats(g, s, cue[fc]) if fc.any() else (np.nan, np.nan)
    return {"label": label, "animal": an, "epoch": epochs.epoch_of(label), "n_trials": int(ok.sum()),
            "push_frac": float(push.mean()), "push_rate_min": _bouts(push) / (len(g) / FS / 60.0),
            "push_amp": float(-s[push].mean()) if push.any() else np.nan,
            "run_frac": float(np.mean(s > RUN_ON)), "cue_push_frac": cp, "cue_dip": cd,
            "cue_push_frac_far_contra": cpf, "cue_dip_far_contra": cdf, "n_far_contra": int(fc.sum())}


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    cur = set(ak.curated_labels())
    labels = [x for x in sorted(ce.sessions_available()) if x in cur and x[-4:] >= "0806"]   # August-only pre
    got, fail = ak.fan_sessions(labels, job, jobs=a.jobs, label="push")
    S = pd.DataFrame([r for _, r in got]).sort_values("label")
    dst = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc" / "push_behaviour_sessions.csv"
    assert_writable(dst)
    S.to_csv(dst, index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    cols = ["push_frac", "push_rate_min", "push_amp", "run_frac", "cue_push_frac", "cue_dip", "cue_push_frac_far_contra",
            "cue_dip_far_contra"]
    order = ["pre", "acute", "subacute", "chronic"]
    A = S.groupby(["animal", "epoch"])[cols].mean().reset_index()
    A["epoch"] = pd.Categorical(A.epoch, order, ordered=True)
    print("per animal x epoch (session means; rule 8 first)\n" + A.sort_values(["animal", "epoch"]).round(3).to_string(index=False))
    if fail:
        print("failed:", [f[0] for f in fail])
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

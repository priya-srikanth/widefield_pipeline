"""Per-trial scaffold (label, trial_id, position, cls, t_min, animal, epoch) for s0 / s1 / s4 / s5 / s6.

Default (unchanged): the AR-HMM collector's trial table, `behavior_states.load_collected()["trials"]`. A run without
the collector's tables (11-22: the adopted four-camera run lp_r4_heatmap_20261013__clean_v37_sync60 has no arhmm/
tree) falls back to the same per-trial columns rebuilt the collector's way (`behavior_states.session_readouts`):
`trial_classes.session_trials(label)` of the run, t_min = (cue - the session's first cue) / 60, for every session of
the production GLM-HMM posterior table. The collector's treadmill / syllable columns (push, walk, lick_period, resp:*)
are not rebuilt; only the composite (TIM) branch of s4 / s5 reads them. The fallback is written once to
``<out_dir>/trial_scaffold.csv.gz`` and reused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["label", "trial_id", "position", "cls", "t_min", "animal", "epoch"]


def _one(label):
    from wfield_local import trial_classes as tc
    T = tc.session_trials(label)
    if T is None or not len(T):
        return None
    T = T.sort_values("trial_id").reset_index(drop=True)
    c0 = float(np.nanmin(T.cue_daq.to_numpy(float)))
    return pd.DataFrame({"label": label, "trial_id": T.trial_id.astype(int).to_numpy(), "position": T.position.to_numpy(),
                         "cls": T.cls.to_numpy(), "t_min": (T.cue_daq.to_numpy(float) - c0) / 60.0})


def glmhmm_labels() -> list[str]:
    from wfield_local import glm_hmm as gh
    return sorted(pd.read_csv(Path(gh.out_dir()) / "states_pooled_all_K5_v2.csv.gz", usecols=["label"]).label.unique())


def collected_available() -> bool:
    from wfield_local import behavior_states as B
    return (B.out_dir() / "trials.csv.gz").exists()


def trials(out_dir: Path | str | None = None) -> pd.DataFrame:
    from wfield_local import behavior_states as B
    if collected_available():
        return B.load_collected()["trials"]
    f = Path(out_dir) / "trial_scaffold.csv.gz" if out_dir else None
    if f is not None and f.exists():
        return pd.read_csv(f)
    print(f"trial_scaffold: no collector table at {B.out_dir()}; rebuilding trials from trial_classes.session_trials",
          flush=True)
    # serial on purpose: s0 / s4 / s5 / s6 are module-level scripts (no __main__ guard), so a spawn pool would re-run
    # them in every worker; the s0 loop over the same sessions is serial too
    T = pd.concat([r for r in (_one(lab) for lab in glmhmm_labels()) if r is not None], ignore_index=True)
    T["animal"] = T.label.str.split("_").str[0]
    T["epoch"] = T.label.map(B.session_epoch)
    T = T[COLS]
    if f is not None:
        tmp = f.with_name(f.name + ".tmp")
        T.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(f)
    return T

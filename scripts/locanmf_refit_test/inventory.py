"""Inventory of cohort-1 sessions for the LocaNMF refit-vs-projection test (2026-11-03).

Read-only. Lists label, epoch, frame count, block_size_max, swapped-pair count, and whether the stored
per-session LocaNMF and the swapmask variant SVTcorr exist.
"""
import json
from pathlib import Path

import numpy as np

from wfield_local import config, epochs
from wfield_local.block_ids import block_size_max_for
from wfield_local.locanmf_cue_lick_analysis import SESSIONS

VARIANT = "meegkit_hpfit_lag_refitT_swapmask"
bad = {}
for line in open("C:/Users/SabatiniLab/swapmask_rerun_logs/build_summary.jsonl"):
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    bad[r["label"]] = r.get("bad")

for s in SESSIONS:
    if s["label"].split("_")[0] not in ("PS92", "PS93", "PS94", "PS95"):
        continue
    lab = s["label"]
    ld = Path(config.locanmf_dir(s["mc"]))
    summ = ld / f"{lab}_locanmf_summary.json"
    ncomp = json.loads(summ.read_text())["n_components"] if summ.exists() else None
    sv = Path(config.svtcorr_path(s["mc"], VARIANT))
    T = np.load(sv, mmap_mode="r").shape[1] if sv.exists() else None
    print(f"{lab}\t{epochs.epoch_of(lab)}\tT={T}\tbsmax={block_size_max_for(s)}\tbad={bad.get(lab)}"
          f"\tncomp={ncomp}\tvariant={sv.exists()}", flush=True)

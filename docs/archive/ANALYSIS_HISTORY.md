# Analysis history — dated incident & status records (archived)

Historical snapshots pruned from `DECISIONS.md` during the 2026-08-09 merge. **Not current** — these record
early-June state and one-off incidents, preserved for provenance. For current decisions/findings see the root
`DECISIONS.md`; for what to run, `runbooks/`.

## Early per-session table (6/1–6/3, as of 2026-06-03)

| Session | FOV (native) | Relabel | Mapping | DAQ h5 | Notes |
|---|---|---|---|---|---|
| 6/1 PS94 | 540×640 full | no | A (raw//2) | `PS94_baseline_20260601_141642.h5` | "baseline"-named but contains task cue/strobe/lick |
| 6/1 PS95 | 540×640 full | no | A (raw//2) | `PS95_baseline_20260601_153627.h5` | same |
| 6/2 PS92 (`illuminated_rescue`) | 487×480 ROI | yes (offset 1) | B | `PS92_20260602_152607.h5` | **functional-channel swap fixed**: use `SVTcorr.npy` (the `*_functional1_WRONG.npy` are bad) |
| 6/3 PS92 | 477×464 ROI | yes (offset 0) | B | `PS92_20260603_104607.h5` | functional channel assumed correct via DAQ relabel |
| 6/3 PS94 | 462×464 ROI | yes | B | `PS94_20260603_175946.h5` | SVD pending at time of writing |
| 6/3 PS95 | 462×464 ROI | yes | B | `PS95_20260603_194442` | motion+SVD pending at time of writing |

Early-June scope note (from the analysis working notes): 6 sessions PS92/PS94/PS95 × {6/1 full-FOV A, 6/2 &
6/3 cleanpairs B}; **6/1 and 6/2 were noisy — early analysis was built on the 6/3 sessions**. DAQ h5 for
6/2,6/3 lived in `<YYYYMMDD>` subfolders with a **2025 year-typo** (`20250602`, `20250603`; also `20250604`).
*(Since fixed: those three dirs were renamed to `2026*` on MICROSCOPE on 2026-08-09 and `sessions.yaml`
updated to match — the paths named here are historical.)*

**Still-to-verify at the time:** 6/3 PS92 / PS95 functional-channel identity (6/3 PS94 was verified correct).

## Decomposition decision, as first framed (SVD now; PMD/LocaNMF "not yet")

The original decision was to **stay on SVD + atlas** for evoked maps and within-animal work, and add LocaNMF
only when moving to cross-animal / subnetwork analysis (this rig had no CUDA GPU; LocaNMF ran on the NeuroCAAS
cloud or a GPU box). **This has since happened** — LocaNMF is now the standard basis for the decode/encode/RSA
analyses (see `DECISIONS.md` Part II). Kept here as the record of when/why the adoption was gated. Reference:
PMD (Couto et al., *Nat Protoc*, PMC8788140); LocaNMF (Saxena et al., *PLoS Comput Biol* 2020, pcbi.1007791).

## Outline rendering fix (atlas overlay)

Region outlines are drawn by the shared `wfield_local/atlas_overlay.region_edges`. The earlier per-module
version marked only the upper/left pixel of each label transition then masked to labeled pixels, dropping the
brain's **left and anterior outer borders** (the open left-anterior / olfactory-bulb edge). The shared version
marks both pixels of each transition before masking, so the outline closes all around (verified left border
96/524 → 524/524).

## Data lifecycle, archival & deletions (2026-06-04)

Three storage tiers; new analysis outputs go to N: (`…\Priya\…`). Raw `.dat` → M: standby, verified, then
deleted from E: (~648 GB freed). Analyzed (motion-corrected `.bin`, SVD/alignment, maps, QC, decks) → N:
`…\Widefield\labcams`, verified (0 missing). DAQ `.h5` → N: `…\DAQ_recorder_output`, verified, then deleted
from E: (4.5 GB). Also deleted from E: after verification: the corrected `.bin` (~621 GB, on N:) and the
**cleanpairs movies** `*_cleanpairs_*_uint16.dat` (~340 GB, regenerable from the M: raw; intentionally NOT
archived). Kept on E: SVD/alignment/maps/QC outputs + the small `*_cleanpairs_frame_map.npz/.csv` +
`*_cleanpairs_summary.json` (needed for regime-B alignment; also on N:).

## 2026-06-04 session issues (PS92 split, PS94 freeze, VS Code auto-update)

**Root cause (both):** VS Code Stable **auto-update** (`CodeSetup-stable-<hash>.exe`, an Inno Setup
installer). If labcams + the DAQ recorder are launched from VS Code's integrated terminal, a VS Code update
restart kills those child processes. Mitigation: launch labcams/DAQ from a **standalone terminal / the `.bat`
launchers**, and set VS Code `"update.mode":"none"`.

**PS94 6/4:** the DAQ "freeze" was display-only — `sample_index_is_contiguous=True`, `recording_complete=True`,
`closed_at` matches the 77.5-min sample count. NO dropped samples.

**PS92 6/4 (split — needs concatenation, DEFERRED):** part1 (`PS92_20260604_133714.h5` +
`…\raw_widefield_data\…`) = 30.0 min then force-closed (`recording_complete=False`); part2 (resumed,
`PS92_20260604_140742.h5` + `…\raw_widefield_data_2\…`) = 41.3 min clean; **~27 s unrecorded gap** between
parts (camera+DAQ off) while the behavior box ran continuously. Concatenation plan (deferred — camera `.dat`
~84 GB): (1) byte-append part1+part2 camera `.dat`; (2) concatenate DAQ analog + digital with the boundary
sample index + gap recorded in attrs; (3) relabel the combined movie; (4) align each part to the continuous
behavior session via shared DAQ events + `created_at`, accounting for the gap offset on part2. *6/4 is excluded
from the curated analysis set, so this remains deferred.*

## Early lick/cue exploratory module inventory

Superseded exploratory modules from the first analysis pass (`wfield_local/`): `locanmf_lick_aligned.py`,
`locanmf_cards.py` / `locanmf_lick_cards.py`, `locanmf_lick_pool.py`, `locanmf_event_master.py`,
`locanmf_cue_lick_analysis.py` (holds the shared legacy `SESSIONS` config), `locanmf_cue_auc.py`,
`locanmf_firstlick_aligned.py`, `locanmf_crossanimal_dff.py`, `locanmf_dff_by_position.py`,
`locanmf_contralateral.py`, `locanmf_encoding_model.py`. The current, load-bearing modules are listed in
`DECISIONS.md` (Module & output reference).

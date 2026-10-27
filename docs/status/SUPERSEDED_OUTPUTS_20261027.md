# Superseded outputs that can be deleted LATER (Priya 10-27)

Priya: "we should mark that we can later delete the superseded figures and outputs from outdated models and analysis".
CLAUDE.md rule 1 (never delete on MICROSCOPE without Priya's permission). 10-27 ~21:00 Priya: "you can delete the superseded
and incorrect files and directories on M - ask me if you are unsure" -> DELETED (5.7 GB + small files): sessions_roi3/,
sessions_roi4/, sessions_roi8/ (env-less), roi2_envfix_20261027/, zscore_check/{session_sd_roi2, session_sd_split_roi2,
session_sd_split_roi2_alm50_tj60_unused, session_sd_roi4_rimlat, session_sd_split_roi4_rimlat}.csv, session_levels/
{patterns, patterns_gap2}.pkl, every v1 `_areas_split*` output (lick_order_*, matched_gauss_*; superseded by areas_splitr2),
offset_contrib/{maps,summary}_areas_split.*. Rows marked DELETED below; the rest wait for their condition.
All paths are derived outputs (reproducible from code + inputs), under
`E = M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/`.

## ALM / tjM1 ROI versions (`scripts/iti_licks.py` `_ROI`; FINAL = v8, DECISIONS.md 10-27 late)

| Path (under E) | Size | What | Delete when |
|---|---|---|---|
| DELETED `sessions_roi3/` | 367 MB, 30 files | v3 (front-edge slivers), killed mid-run | now |
| DELETED `sessions_roi4/` | 1.3 GB | v4 (lateral rim, detached ALM corner); extracted WITHOUT the frozen FR env | now |
| DELETED `sessions_roi8/` | ~1.3 GB | v8 first run WITHOUT the frozen env (complete, FaceRhythm fields empty); env-correct files are in `sessions_roi8fr/` | now |
| `sessions_roi2/` | 1.4 GB | v2 (ALM 40 / tjM1 50, 16-px rim; env-fixed) | once v8 is the default and its redo is checked |
| DELETED `roi2_envfix_20261027/` | 2.7 GB | v2 env-fix work dir + backup of the defective v2 files | now (fixed files are in sessions_roi2/) |
| `roi_v1_backup_20261027/` | 2.0 GB | copies of the v1 (40/40) CSVs / figures / figdata before the v2 redo | after the v8 numbers are on the page and checked against it |
| `sessions/` | (large) | v1 (40/40) session files | keep while v1 is the reference for any published number; then optional |
| DELETED `zscore_check/session_sd_roi2.csv` | 4 B | empty file from the failed first v2 run | now |
| DELETED `zscore_check/session_sd_split_roi2.csv` | | v2 split SDs with ALM 50 / tjM1 60 (never used) | now |
| DELETED `zscore_check/session_sd_split_roi2_alm50_tj60_unused.csv` | | copy of the same | now |
| DELETED `zscore_check/session_sd_roi4_rimlat.csv`, `session_sd_split_roi4_rimlat.csv` | | v4 SDs | now |
| `zscore_check/session_sd_roi2_tj50.csv`, `session_sd_split_roi2_tj50.csv` | | v2 SDs | with sessions_roi2/ |
| DELETED outputs tagged `_areas_split` (no r2) | | split decoder on v1 ROIs (`lick_order_preds_areas_split*`, `matched_gauss_areas_split*`) | now (r2 / v8 versions supersede) |

## Per-session map caches (`session_levels/`)
| DELETED `patterns.pkl` | 4 MB | built with the old 1-s ITI first-lick rule | now |
| DELETED `patterns_gap2.pkl` | 4 MB | lacks encoding slopes / fit shares | now |

## Decoder / basis runs superseded by decisions
| outputs tagged `_svd` (lick-window scaled) and `_svdraw` | SVD scaling checks; frame-scaled `_svdF` supersedes | after the basis decision is written up |
| outputs tagged `_locanmf` (lick-window scaled, unmasked) | `_locanmfF` / `_locanmfFM` supersede | after the basis decision |
| `iti_lick_order` outputs with the retired 'beyond the pull' metric | read-out is now same-place (gauss) | when figures no longer cite them |

## Figures
| `figures/` PNGs whose stem is not on the goal page and not produced by a current script (e.g. `iti_*` names renamed to `lick_*` on 10-27) | after a page audit (list with `grep src= _goal_page/index.html`) |
| `_goal_page/` local PNGs no longer referenced by index.html | after the v8 publish |

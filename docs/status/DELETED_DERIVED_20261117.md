# Deleted derived outputs, 2026-11-17

Priya's decision 11-17 (relayed by the coordinator): the four frozen lick-window epoch decoders written into the rebuild
store by the 11-16 15r LP-lick render are incorrect derived outputs. That render's first attempt imported the STALE main
checkout (C:/Users/SabatiniLab/Github/widefield_pipeline, editable install; its log printed REFUSED from
render_profile.code_check and continued) and froze the models ("frozen-new"); the second attempt (wf_rebuild code)
only served them back ("frozen-hit"). Deleted so the rebuild-code render refits them (docs/FROZEN_MODELS.md §8; frozen
models are references, deleted here only by Priya's explicit permission for incorrect derived outputs).

Checked before deleting: `find <store> -newermt 2026-11-16` lists exactly these four model directories (plus their
parent animal directories' mtimes); no other file of the same stem exists. Each directory held manifest.json + model.joblib.
Every one has `lick_source: lp`, `mask_sig 5f9eb36f36b2`, freeze_version 2, run selection
sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70. The older DAQ-arm siblings of the same
basis (11-05) are KEPT.

Store: M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/frozen_models/

| path (relative to store) | bytes | mtime (local) | frozen_utc | train_fp | n_engaged_pre / n_pre_sessions |
|---|---|---|---|---|---|
| PS92/epoch_decoder_lick_joint76d884_a07055c3580c/model.joblib | 499572 | 2026-11-16 17:51:16 | 2026-11-16T22:51:15.314815 | 8d7ef9d0dcab2b32 | 2863 / 8 |
| PS92/epoch_decoder_lick_joint76d884_a07055c3580c/manifest.json | 1863 | 2026-11-16 17:51:16 | 2026-11-16T22:51:15.314815 | 8d7ef9d0dcab2b32 | 2863 / 8 |
| PS93/epoch_decoder_lick_joint0141bb_a260f0abdbbe/model.joblib | 407284 | 2026-11-16 17:54:35 | 2026-11-16T22:54:34.021440 | 1c2f4b0283ade320 | 3668 / 7 |
| PS93/epoch_decoder_lick_joint0141bb_a260f0abdbbe/manifest.json | 1762 | 2026-11-16 17:54:35 | 2026-11-16T22:54:34.021440 | 1c2f4b0283ade320 | 3668 / 7 |
| PS94/epoch_decoder_lick_jointa5e70d_cbfeae0130c3/model.joblib | 473652 | 2026-11-16 17:56:42 | 2026-11-16T22:56:41.844829 | 8e9396339f7fd278 | 4162 / 8 |
| PS94/epoch_decoder_lick_jointa5e70d_cbfeae0130c3/manifest.json | 1859 | 2026-11-16 17:56:42 | 2026-11-16T22:56:41.844829 | 8e9396339f7fd278 | 4162 / 8 |
| PS95/epoch_decoder_lick_jointa91701_79a6814f01b4/model.joblib | 499572 | 2026-11-16 17:58:46 | 2026-11-16T22:58:45.646191 | 140fd93fc2f5f081 | 4748 / 8 |
| PS95/epoch_decoder_lick_jointa91701_79a6814f01b4/manifest.json | 1858 | 2026-11-16 17:58:46 | 2026-11-16T22:58:45.646191 | 140fd93fc2f5f081 | 4748 / 8 |

Deleted 2026-11-17 10:14 (rm of the 8 files, rmdir of the 4 directories); verified gone afterwards. The four 11-05
DAQ-arm siblings remain (epoch_decoder_lick_joint76d884_0b48db98230e, _joint0141bb_f9ef082366ff, _jointa5e70d_d5211f7a2aa7,
_jointa91701_017f6aacb806).

## Refit check (11-17, render from wf_adeck1117 = origin/main 36cabf95 + this branch, PYTHONPATH set)

The 15r LP-lick render from this branch refitted all four (log: "frozen-new", 10:25-10:41). Each refit has the SAME
spec_id, the SAME train_fp (8d7ef9d0dcab2b32 / 1c2f4b0283ade320 / 8e9396339f7fd278 / 140fd93fc2f5f081), the same
n_engaged_pre (2863 / 3668 / 4162 / 4748) and the same model.joblib size as the deleted files. The training rows were
therefore identical: the deleted models were reproducible by the current code (the 11-16 main-checkout code and the
current code select the same pre-stroke training trials for this kind). The gate description in the log ("camera gate:
LP-attempted trials: every camera-seen lick") is the same in both; the wf_rebuild run of 11-16 printed "DAQ lick-arm
trials" while hitting the same ids, i.e. only its label text differs.

## 11-17 ~15:30 — RESTnlc1 v1 timing-run caches (Priya OK 11-17: "ok to delete the orphan cache files")

Written 14:31-14:33 by a one-session timing run under the superseded post-cue cam1 definition (lp_lick_times); never read by any code (v2 uses session_cache_restnlc1v2 / restw_check_c1v2).
- `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/session_cache_restnlc1/PS92_0814__restnlc1-hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk_csig-3d7bbe921e__44b2de7cd98d8e89.npz` — 1058631 bytes, mtime 2026-11-17 14:32:17
- `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/restw_check_c1/sessions/PS92_0814.npz` — 30106 bytes, mtime 2026-11-17 14:33:20.0

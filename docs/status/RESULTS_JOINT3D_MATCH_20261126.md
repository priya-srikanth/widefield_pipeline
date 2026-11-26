# Joint 3-D like-for-like match: results (CC-B, 2026-11-26)

Priya 11-26: "sure let's try the joint 3d match".

- Branch: `ccb/joint3d-match-1126` (worktree `Github/wt_joint3d_ccb1126`). Not pushed.
- Script: `scripts/joint3d_1126/lick_matched_joint3d.py`.
- Run: 11:07-12:14 EST.
- Folders:
  - this record folder;
  - a copy of the tables in `rebuild_20261123_fourcam/run_records_joint3d_match_ccB1126/`.

## Question

The page builds each like-for-like offset on one axis only:

- Fig 12.1 matches on lateral;
- Fig 11.1 matches on forward and down;
- Fig 11.10 matches on angle and reach.

So could the lateral offset partly reflect post-stroke licks landing at a different forward or down place than their lateral-matched pre references?

Here the reference weight is a 3-D Gaussian on the (lateral, forward, down) landing, with a diagonal covariance and one σ per axis in mm. With it, the lateral (primary), forward and down offsets are recomputed. Everything else follows the page's recipe:

- **Reference:** the animal's PRE licks of the same lick set, any spout. Pre sessions are scored leave-one-session-out.
- **Mean:** Nadaraya-Watson mean of the stored per-lick read-outs of the frozen pre-stroke lick readers. No re-extraction and no decoder fit.
- **Sessions:** session means over ≥ 8 licks.
- **Statistics:** change from the animal's own pre, by the 95% paired nested bootstrap (animals → sessions, `analysis_kit.boot_delta`, rng 0). This runs through the same `lick_matched_readout.sessions` / `.tests` functions as the page.
- **Correction:** none, as on Fig 12.1.

### Inputs (read-only)

The run used the lane E reader inputs of run clean_v37_sync60:

- **Patch folder:** `iti_licks_patched_fusedorder_laneE_20261124`.
  - `lick_order_preds_areas_splitr8.csv.gz` holds Fig 12.1's per-lick lateral read-out and landing.
  - `where_3axis_preds_splitr8.csv.gz` holds the 3-D landing and the forward / down read-outs of Fig 11.1.
  - `iti_dock_audit.csv` holds the docked flag `iti_after_quiet_1s`.
- **Licks:** the fused four-camera lick list.
- **3-D landing:** `pose3d_4cam_3net_20261121_sync60/kinematics_fused/licks.csv.gz`. All 245,327 landings in the 3-axis cache were found as rows of this table, matched to 3 decimals.

The two per-lick caches were paired row by row:

- 0 of 245,304 licks were left unpaired.
- The two lateral read-outs agree exactly on cued licks and to within 0.13 mm on ITI licks.

### Lick sets

The four lick sets are the page's:

| Lick set | Definition |
|---|---|
| ITI docked bout starts | `WFIELD_ITI_FILTER=iti_after_quiet_1s`: the ITI lick that starts a bout after ≥ 1 s without licking, spout docked. This is the Fig 12.1 headline. |
| ITI all | Every ITI lick. |
| Cued | Post-cue licks of attempted trials. |
| Cued first lick | The first post-cue lick of an attempted trial. |

## 1. Reproduction check: PASSED

The 1-D lateral reference was run through this code. Both production reader folders were reproduced to float round-off:

- `iti_licks_readers2_laneE_20261124` (all ITI);
- `iti_licks_readers2_dock_laneE_20261124` (docked).

| Run | `matched_gauss_tests.csv` | `matched_gauss_sessions.csv` | `matched_gauss_sigma_cv.csv` | σ chosen |
|---|---|---|---|---|
| All ITI | 378 rows, max \|Δ\| 4.4e-16 | 4,399 rows, 4.4e-16 | 6 rows, 1.1e-16 | 0.1 mm |
| Docked | 315 rows, 4.4e-16 | 3,386 rows, 4.4e-16 | 6 rows, 1.1e-16 | 0.1 mm |

The docked run kept 11,449 of 84,804 ITI licks, the same count as the production log.

The forward / down 1-D numbers below use the page's σ (forward 0.1 mm, down 0.2 mm). On all ITI and cued licks they match the published `matched_axes_tests.csv` to within about 0.002 mm. For example, ITI forward acute is −0.113 here and −0.113 published, and cued down subacute is −0.199 here and −0.201 published. They do not match exactly because that script pairs licks on the 3-axis cache rows.

The page has no docked forward / down numbers. The docked 1-D forward / down values in the table are therefore new.

## 2. σ chosen by held-out pre prediction

The kernel is diagonal. The per-axis σ grid was:

- lateral: 0.05, 0.1, 0.15, 0.2, 0.3 mm;
- forward and down: 0.1, 0.2, 0.3, 0.5, 1.0, 2.0 mm.

The 2.0 mm setting effectively switches that axis off, so the grid contains the 1-D match.

The selection rule had three parts:

- **Error:** held-out PRE squared error, leave-one-pre-session-out, pooled over lick sets and animals. This is the same rule as the 1-D σ.
- **Candidates:** only grid points that give ≥ 95% of pre licks a reference.
- **Comparison set:** the error is compared on the pre licks that every candidate covers, so a narrow kernel cannot win by skipping hard licks.

The choice was made separately for each run, as on the page.

| Target | All-ITI run (lat / fwd / down, mm) | Docked run (lat / fwd / down, mm) |
|---|---|---|
| Lateral (primary) | **0.2 / 0.2 / 0.3** | **0.15 / 0.2 / 0.3** |
| Forward | 0.2 / 0.2 / 0.3 | 0.2 / 0.2 / 0.3 |
| Down | 0.3 / 0.2 / 0.2 | 0.2 / 0.2 / 0.2 |

The forward / down landing carries information about the lateral read-out. On the same held-out pre licks, the lateral error is:

| Run | Joint kernel | Near-1-D kernel (lateral 0.1, forward / down 2.0) | Change |
|---|---|---|---|
| All ITI | 0.205 | 0.221 | −7% |
| Docked | 0.186 | 0.207 | −10% |

Table: `joint_sigma_cv.csv`.

The sensitivity run used a fixed σ of 0.1 mm on all three axes.

## 3. Primary cells: lateral offset, per-axis vs joint

The offset is the change from the animal's own pre, in mm, + = contralesional. Brackets give the 95% paired nested bootstrap CI. * = the CI excludes 0. k/n = the number of animals that moved in the direction of the mean, out of the analyzable animals.

### ITI docked bout starts, lateral (Fig 12.1 headline)

| Epoch | Per-axis (1-D, σ 0.1) | Joint 3-D (σ 0.15 / 0.2 / 0.3) | Joint, fixed σ 0.1 |
|---|---|---|---|
| Acute | +0.28 [+0.06, +0.52]* 4/4 | +0.15 [+0.01, +0.31]* **3/3** | +0.10* 1/1 |
| Subacute | +0.17 [−0.01, +0.38] n.s. 3/4 | +0.21 [+0.04, +0.39]* **3/3** | +0.06 n.s. 1/1 |
| Chronic | +0.22 [+0.08, +0.45]* 4/4 | +0.25 [+0.09, +0.53]* **3/3** | +0.02 n.s. 1/2 |

Per animal (acute / subacute / chronic):

| Animal | 1-D | Joint |
|---|---|---|
| PS92 | +0.07 / +0.19 / +0.10 | +0.04 / +0.18 / +0.13 |
| PS93 | +0.06 / +0.02 / +0.10 | +0.07 / +0.03 / +0.10 |
| PS94 | +0.42 / +0.47 / +0.56 | +0.32 / +0.41 / +0.54 |
| PS95 | +0.59 / −0.01 / +0.13 | not analyzable |

**PS95 drops out of the docked set under the joint match.** PS95 has only 178 docked pre licks. 175 of them, and 844 of 864 chronic licks, have no 3-D neighbor among its other pre sessions, so no session keeps 8 licks.

### Cued FIRST lick, lateral

| Epoch | Per-axis (1-D, σ 0.1) | Joint 3-D (σ 0.2 / 0.2 / 0.3) | Joint, fixed σ 0.1 |
|---|---|---|---|
| Acute | +0.32 [+0.27, +0.38]* 4/4 | +0.18 [+0.09, +0.25]* 4/4 | +0.16 [−0.03, +0.33] n.s. 3/4 |
| Subacute | +0.19 [+0.13, +0.26]* 4/4 | +0.11 [+0.03, +0.21]* 4/4 | +0.08 n.s. 3/4 |
| Chronic | +0.12 [+0.05, +0.18]* 4/4 | +0.06 [+0.02, +0.11]* 4/4 | +0.05 n.s. 4/4 |

Per animal, acute:

| Animal | 1-D | Joint |
|---|---|---|
| PS92 | +0.29 | +0.09 |
| PS93 | +0.29 | +0.24 |
| PS94 | +0.36 | +0.21 |
| PS95 | +0.34 | +0.18 |

### Other lick sets, lateral

| Lick set | 1-D (acute / subacute / chronic) | Joint (acute / subacute / chronic) |
|---|---|---|
| ITI all | +0.36* / +0.24* / +0.27*, all 4/4 | +0.30* / +0.20* / +0.24*, all 4/4 |
| Cued all | +0.18* 4/4 / +0.03 n.s. / −0.03 n.s. | +0.12* 4/4 / +0.01 n.s. / −0.04 n.s. |

### Forward and down

- **Cued forward:** the read-out lies behind the landing in all epochs and survives the joint match. It is −0.16* / −0.20* / −0.16* with 1-D and −0.12* / −0.18* / −0.15* joint, all 4/4.
- **Cued down:** stronger with the joint match. Subacute goes from −0.20* to −0.28* and chronic from −0.11 n.s. to −0.18* (4/4).
- **ITI-all forward, acute:** −0.11* (4/4) becomes −0.09 n.s. (3/4).
- **Cued first-lick forward, acute:** −0.09* becomes −0.03 n.s.

Full table: `side_by_side.csv`, which lists every lick set, target, epoch and kernel with CI, k/n and per-animal values. All scopes, including per spout and first / later licks, are in `tests_all.csv`.

## 4. n_eff and dropped licks (lateral target, joint kernel)

### How licks lose their reference

A lick gets no reference when its total kernel weight is < 3 (the page's rule) or when n_eff < 5 (Kish). n_eff < 5 alone removed almost nothing: 0-4 licks per set. The weight rule did the dropping.

### Share of licks without a reference

| Lick set | Joint | Fixed σ 0.1 |
|---|---|---|
| ITI docked | **45%** | 91% |
| ITI all | 12.5% | 58% |
| Cued all | 6.6% | 37% |
| Cued first lick | 14% | 55% |

The 1-D reference covers 89-100% of licks per set × epoch.

### Per animal × epoch

Each cell is the median n_eff, then (licks without a reference / licks).

| Lick set | Animal | Pre | Acute | Subacute | Chronic |
|---|---|---|---|---|---|
| ITI docked | PS92 | 19 (231/941) | 12 (259/570) | 12 (89/185) | 19 (306/1186) |
| ITI docked | PS93 | 51 (121/1599) | 9 (316/519) | 15 (183/391) | 17 (670/1564) |
| ITI docked | PS94 | 12 (471/1239) | 2 (287/328) | 7 (393/649) | 6 (696/1064) |
| ITI docked | PS95 | 3 (175/178) | 1 (23/23) | 2 (145/149) | 3 (844/864) |
| ITI all | PS92 | 111 (275/8622) | 79 (312/2952) | 75 (117/1205) | 129 (254/8095) |
| ITI all | PS93 | 192 (197/8138) | 66 (561/2989) | 90 (275/2626) | 96 (1009/10710) |
| ITI all | PS94 | 96 (414/10740) | 30 (1033/3544) | 72 (753/6108) | 83 (765/8612) |
| ITI all | PS95 | 11 (619/1353) | 7 (140/232) | 9 (737/1342) | 13 (3181/7536) |
| Cued all | PS92 | 235 (190/12000) | 153 (505/7500) | 179 (278/3000) | 283 (304/18000) |
| Cued all | PS93 | 186 (138/10500) | 172 (676/6000) | 170 (270/4500) | 196 (805/18000) |
| Cued all | PS94 | 272 (293/12000) | 26 (3331/9000) | 185 (624/7500) | 143 (892/12000) |
| Cued all | PS95 | 183 (214/12000) | 138 (143/1500) | 131 (1245/10500) | 164 (677/16500) |
| Cued first | PS92 | 158 (28/996) | 99 (109/945) | 108 (24/243) | 141 (40/1412) |
| Cued first | PS93 | 148 (22/1081) | 39 (240/818) | 76 (66/416) | 100 (134/1351) |
| Cued first | PS94 | 192 (45/1257) | 8 (712/1294) | 101 (142/620) | 79 (228/1051) |
| Cued first | PS95 | 88 (34/1602) | 59 (55/213) | 69 (253/1076) | 100 (103/1485) |

Table: `neff.csv`, which covers all targets and kernels, with the 10th percentile of n_eff.

### The drop is not random

It removes post-stroke licks that land outside the animal's pre 3-D landing cloud, mostly acute and mostly PS94. So the joint offset is the offset of licks that landed where pre licks landed in 3-D. Part of its shrinkage on the cued first lick is this selection, not only the removal of a forward / down confound.

## 5. Does any conclusion change?

1. **The lateral contralesional offset is not a forward / down landing artifact.** It survives joint 3-D matching in every set where the page calls it:
   - ITI all: about 15% smaller, all epochs *, 4/4.
   - ITI docked: 3 of 3 analyzable animals, all epochs *.
   - Cued, acute: +0.12*, 4/4.
   - Cued first lick: all epochs *, 4/4.
2. **The cued FIRST-lick lateral offset is about 45% smaller under the joint match**, at +0.18 / +0.11 / +0.06 instead of +0.32 / +0.19 / +0.12, and it is still significant with 4/4 animals. About half of the per-axis first-lick offset goes with forward / down landing differences, or with licks outside the pre 3-D cloud. Pre-matched first licks are still read contralesionally. If the page quotes the first-lick magnitude, it should carry the joint number as a sensitivity line.
3. **The docked ITI headline cannot be checked at 4/4 in 3-D.** PS95's docked pre set is too sparse: 178 licks, median n_eff 3. With the other three animals, the joint offset is significant in all epochs, including subacute, which is n.s. with 1-D. The joint match does not weaken the docked result, but it rests on 3 animals.
4. **Forward / down:**
   - The cued forward deficit (read-out behind the landing) holds.
   - Cued down becomes stronger subacute and chronic.
   - The acute ITI and cued first-lick forward offsets lose significance under the joint match.
5. **The fixed σ 0.1 mm on all axes is too sparse to read.** Median n_eff is 1-3 for ITI, 37-91% of licks have no reference, and the docked set keeps 1-2 animals. Its point estimates have the same sign and a similar size as the joint ones. Its loss of significance is a coverage effect and does not count as evidence against the offset.

## 6. Files

- `side_by_side.csv`: the per-axis vs joint vs fixed-σ table for all four lick sets.
- `tests_all.csv`: every scope.
- `sessions_<run>_<target>_<kernel>.csv`: the session values.
- `neff.csv`: n_eff and dropped licks.
- `joint_sigma_cv.csv`: the σ grid.
- `run_meta.json`: input sha256 / size / mtime, the reproduction check and the chosen σ.
- `joint3d_vs_1d_offset.png` (+ `svg/`): figure in the style of Fig 12.1.

### Figure

- Rows are the four lick sets. Columns are lateral, forward and down, each with 1-D then joint.
- Small filled markers are sessions and open markers are animal means. Shape = animal (circle PS92, square PS93, triangle PS94, diamond PS95) and color = epoch. The bar is the mean of animals.
- Stars and k/n sit in the top band. There is no top spine and no lines across epochs.
- Caption details:
  - **Window:** the per-lick read-out of the frozen lick readers.
  - **Licks:** the fused four-camera list.
  - **Reference:** pre licks of the same set, leave-one-session-out.
  - **σ:** as titled on each panel.
  - **Baseline:** the animal's own pre.
  - **Correction:** none.

## 7. Process

- **Writes:** the compute ran with `WFIELD_RUN_OUT` set to this folder and wrote nothing else.
- **First launch:** it was refused by the protected-roots fence when it tried to write into `rebuild_20261123_fourcam/` without `WIDEFIELD_REBUILD_ROOT`. Nothing was written there except an empty owner marker. The tables were then copied in by `--copy-to`, run with `WIDEFIELD_REBUILD_ROOT` set.
- **Frozen stores:** load-only (`WIDEFIELD_FROZEN_LOAD_ONLY=1`); no decoder was fit or frozen.
- **Processes:** only PIDs 5604 (failed first launch) and 15548 were started, both CREATE_NO_WINDOW.
- **Protected-mtime audit** (`tools/protected_mtime_audit.py --since "2026-11-26 10:31"`, run 12:20):
  - Every hit was in this run's own declared folder or in other windows' concurrent runs.
  - Other windows' hits in `rebuild_20261123_fourcam`: `o2/o2f_spd_20261125`, `run_records_cd_overlap_null_ccB1126_chronic`, `figures_working/ccB_nosejaw_traj_20261126` and `ccB_spdcompare_20261126`.
  - Other windows' hits in `vls_stroke_paper`: `ccB_fr5b_allspouts_2026-11-26`, `ccB_render_kits_2026-11-25`, 9 files in `manuscript_live/figures`, and `CHANGES_SINCE_20261124.md`.
  - There were 0 hits in the LP_analysis production, patch or lane E reader folders this run read, and 0 hits from this run outside its two folders.

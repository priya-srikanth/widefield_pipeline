# STATUS — 2026-10-07: Chronic neural stability → cohort imaging COMPLETE (PS92/PS93/PS94/PS95)

**DECISION (Priya, 2026-10-07): imaging with this cohort is complete.** Last sessions are day ~50
post-stroke (PS92/PS93 on 2026-10-06, PS94/PS95 on 2026-10-05). The evidence below — behaviour and
every neural performance/decode readout plateaued, and the one unsettled signal (crossnobis
distance-from-pre) shown to be representational *drift* rather than ongoing reorganization —
supports ending acquisition. Histology/sac timing is the PI's call; this record documents the
neural + behaviour evidence only. Continues `STATUS_2026-09-28_CHRONIC_STABILITY.md`.

Analysis box (M: = MICROSCOPE). Reasoning + durable pointer: `DECISIONS.md` 2026-10-07 entry (F18).

---

## TL;DR

1. **Performance / decoding have plateaued in all four animals.** Flat (non-rising) across the whole
   cohort: far-contra hit rate, encoder ceiling / frozen / gain, best-match accuracy, both decoders
   (frozen + refit). These are the "has the chronic level settled" metrics.
2. **The only still-moving signal is the representational geometry** — crossnobis distance from each
   animal's own pre-stroke template is still rising in **PS92, PS93, PS95** (`crossnobis mean(6 pos)`),
   plus a few map-amplitude / reorganisation-G series. **PS94 is fully flat.**
3. **That rise is DRIFT, not reorganization.** A fixed pre-template cannot separate the two (both grow
   distance-from-pre over time), so the follow-up `scripts/chronic_drift_vs_reorg.py` instead compares
   the *per-day rate* of session-to-session change in two windows. In all four animals the
   within-chronic rate is **not detectably above each animal's own pre-stroke drift floor** — the
   chronic representation keeps inching away from pre at roughly the baseline pre-stroke drift rate. So
   the rising distance-from-pre does **not** argue for continued recording.

---

## Session counts (stroke_date: PS92/PS93 = 2026-08-17, PS94/PS95 = 2026-08-16)

| Animal | chronic_from (day) | pre sessions (used*) | chronic sessions | chronic day range |
|--------|--------------------|----------------------|------------------|-------------------|
| PS92 | 11 | 11 | **12** | 11–50 (0828–1006) |
| PS93 | 11 | 11 | **12** | 11–50 (0828–1006) |
| PS94 | 25 | 11 | **8**  | 25–50 (0910–1005) |
| PS95 | 15 | 11 | **11** | 15–50 (0831–1005) |

\* "used" = pre sessions passing the ≥10-trials/position gate in `grant_kit._collect_7` (fewer than
the 13–16 registered; the two pre clusters are early June and early August). Chronic counts match the
pooled `n_chronic` in `chronic_stability.csv`.

---

## Evidence 1 — `chronic_stability` (day-50 refreshed, this nightly)

Rule (`epochs._plateau_index`): *flat* = chronic tail not still rising (one-sided drift ≤ K_DRIFT·pre-SD);
*settled* = residual ≤ K_RES·pre-SD. Pooled-pre readouts (behaviour, encoder, best-match, crossnobis,
map amplitude) can't fail "settled" by construction → read *flat*; per-session readouts (decoders,
reorg-G) read *flat + settled*.

- **Flat in all 4:** behaviour far-contra hit, encoder ceiling/frozen/gain, best-match accuracy,
  decoder frozen, decoder refit.
- **Still rising (flat = False):** `crossnobis mean(6 pos)` PS92/PS93/PS95; `crossnobis far-contra`
  PS92/PS95; `reorganisation G` PS92; `map amplitude far-contra` PS95; `map amplitude near-ipsi`
  PS93/PS95.
- **PS94 cleanest** — every metric flat (its `crossnobis mean` went flat at day 50).
- *Settled* is rarely met even where flat → no series is a tight locked plateau; "no longer trending",
  not "pinned".

Artifacts: `M:/MICROSCOPE/Priya/Widefield/labcams/chronic_stability/chronic_stability.{csv,png}`.

---

## Evidence 2 — drift vs reorganization (NEW: `scripts/chronic_drift_vs_reorg.py`)

Per animal, in the same pre-between-position crossnobis units as the 8-series (cue/working):

- `m_pre` = slope of pairwise crossnobis(pre_i, pre_j) vs |day gap|  → the **drift floor**.
- `m_chr` = slope of pairwise crossnobis(chr_i, chr_j) vs |day gap| → the **within-chronic rate**.
- Pairwise distance = mean own-position (diagonal) of `grant_geometry._crossnobis_cross`, averaged over
  15 half-splits, ÷ each animal's pre between-position scale (`_crossnobis_within` on the pooled pre).
- Bootstrap over sessions (n=2000) for slope CIs and Δ = m_chr − m_pre.

| Animal | m_pre /day | m_chr /day | Δ (chr−pre) | Δ 95% CI | verdict |
|--------|-----------|-----------|-------------|----------|---------|
| PS92 | 0.0051 | 0.0110 | +0.0059 | [−0.017, +0.016] | drift-consistent |
| PS93 | 0.0095 | 0.0085 | −0.0009 | [−0.013, +0.008] | drift-consistent |
| PS94 | 0.0026 | 0.0032 | +0.0006 | [−0.013, +0.007] | drift-consistent |
| PS95 | 0.0039 | 0.0106 | +0.0068 | [−0.011, +0.016] | drift-consistent |

**Conclusion: all four drift-consistent** — no animal's chronic representation is changing detectably
faster than its own pre-stroke baseline drift. In the figure, at the overlapping day-gaps (0–40 d) the
pre-pre and chronic-chronic clouds sit on top of each other; **PS94 is flattest**; **PS92 and PS95**
show a marginally steeper chronic slope (point-Δ ~2× baseline) whose CI still spans 0 — the only
candidates for any residual reorganization.

Artifacts: `.../labcams/chronic_stability/chronic_drift_vs_reorg.{csv,png}`.

---

## Caveats / limits (do not over-read)

- **Underpowered.** 8–12 chronic + 11 pre sessions per animal, and pre coverage is sparse with a
  gap between the June and August clusters (pre-pre pairs are small-gap or 60–70 d, nothing between),
  so the drift-floor slope is leveraged by the far cluster while chronic only reaches ~40 d gaps. The
  result is **"no evidence chronic exceeds baseline drift"**, not proof of zero reorganization; it
  cannot rule out a *modest* residual reorg component — exactly where PS92/PS95's point estimates sit.
- Verdict rests on cue/working (the deck's crossnobis readout). Other arms not yet checked for this
  analysis.

---

## Known defect found while checking this (NOT blocking the decision)

The **decoder** readouts in this run's `chronic_stability` silently dropped each animal's **two October
chronic sessions** (PS92/PS93: 1002+1006; PS94/PS95: 1001+1005), so the decoder frozen/refit/G verdicts
were computed on days 11–43 only (n = 10/10/6/9 instead of 12/12/8/11). Cause: `recovery_trajectory`
tags days with `grant_kit._day` (month×31, which runs +1 for October), while `chronic_stability`
(`build_series`, line ~178) joins decode values to sessions by calendar `days_since_stroke` — the
October rows (47/51) don't match the calendar day map (46/50) and are excluded by
`if int(r.day) in d2l[a]`. **Fixed upstream in `660bcbe`** (`grant_kit._day` → calendar diff). The next
`chronic_stability` run on that code will join correctly and restore full decoder coverage. Pooled
readouts (crossnobis/encoder/map/behaviour) are list-order keyed, so they were unaffected and used all
12/12/8/11 sessions.

---

## Outstanding / optional (none blocks the completion decision)

1. **Re-run `chronic_stability` on the `_day`-fixed code** (≥ `660bcbe`) to recompute the decoder
   verdicts over the full chronic set incl. the October/day-50 sessions. Expected to stay flat; worth
   confirming for PS92 (its dropped 1002 session had a `frozen` dip to 0.60 vs ~0.73 norm).
2. Optional refinements to `chronic_drift_vs_reorg`: restrict both windows to the 0–40 d overlap;
   per-position breakdown; other alignment arms.
3. Fold `chronic_drift_vs_reorg` into the nightly/deck if we want it tracked going forward (currently a
   standalone script, committed `c3889bc`, not wired into `nightly_figs`).

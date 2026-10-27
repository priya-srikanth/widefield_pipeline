"""Imaging page, encoder-page format step 2 (Priya 10-26, via e3): a TITLE above every figure (model / decoder, window,
one-line finding), a METHODS block opening every caption (decoder / measure, frozen or refit, window, reference, gate,
sessions / animals, statistic, correction, n) and the script that draws it; sidebar entries use the title.
Methods were traced from the code that writes each source PNG (notes: prov_s0_4.md, prov_s5_13.md, prov_s7.md,
prov_s9.md in this folder). Also corrects caption statements the trace showed to be wrong (fig 0.1, 1.1, 1.2, 4.1, 9.2,
12.1; LP pre-cue window = 1 s, not 2 s). Runs after p_img_renum.py."""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
P = HERE / "imaging_page" / "imaging_results.html"
s = P.read_text(encoding="utf8")
assert 'class="ftitle"' not in s, "already titled"


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# ---------------------------------------------------------------- shared method strings
EPO = ("Epochs per animal from its own lesion day (acute = far-contra hit rate &lt; 25% of pre: PS92 days 1&ndash;5, PS93 "
       "1&ndash;4, PS94 1&ndash;7, PS95 1; chronic from day 11 / 11 / 25 / 15; subacute between)")
NS = "4 animals; sessions pre 44 (June + August, 11 per animal) / acute 16 / subacute 17 / chronic 43"
DEC_F = ("<b>Decoder:</b> FROZEN pre-stroke 6-way multinomial logistic regression (standardised, C = 0.5) on each "
         "animal&rsquo;s joint-LocaNMF dorsal-cortex components in time sub-bins (session z-scored), fit once on all "
         "that animal&rsquo;s pre-stroke engaged trials and never refit; the pre bar is scored leave-one-session-out. "
         "<b>Measured:</b> accuracy of the true spout, chance 1/6.")
ST_ACC = ("<b>Stats:</b> bars = trial-pooled accuracy, dots = sessions; paired nested bootstrap (animals &rarr; sessions "
          "&rarr; position blocks, 2,000 draws); * = 95% CI of epoch &minus; pre excludes 0 (uncorrected), ** = still "
          "excludes 0 after Bonferroni over 18 tests (6 positions &times; 3 post epochs). " + NS + ". " + EPO + ".")
WORK = "working trials = licking trials plus misses while still working (terminal quit period excluded)"
ST_EPO = ("<b>Stats:</b> nested bootstrap (animals &rarr; sessions, 2,000); * = 95% CI of epoch &minus; pre excludes 0 "
          "(uncorrected), ** = after Bonferroni over post epochs &times; positions; no mark with &lt; 2 animals. " + NS + ". ")
LMM = ("<b>Model:</b> linear mixed model, animal random intercept + slope (intercept-only fallback), fit with "
       "<code>wfield_local/mixed_models.py</code>; q (Benjamini&ndash;Hochberg) is in <code>imaging_encoding/qc/imaging_lmm*.csv</code>, "
       "the panels show coefficient and z.")
CD = ("<b>Measure:</b> FROZEN pre-stroke coding direction per spout (mean of trials at P minus mean of the rest, unit "
      "norm, on window-mean joint-LocaNMF components z-scored in the pre-stroke frame; glue / bulb components dropped; "
      "the shared mode projected out). Pole units: 0 = pre-stroke not-P, 1 = pre-stroke P. Success (lick) trials; the "
      "pre-cue window is also lick-free.")
CD_LOO = (" Refit on August pre only (08-06 to 08-14); each pre session is scored leave-one-session-out (its own session "
          "removed from directions, poles, z-frame and shared mode), post sessions on the fit to all August pre. "
          "Readout windows: pre-cue [&minus;2, 0) s, cue [0, 1) s, lick [0, 1) s from the first lick. Sessions pre / acute "
          "/ subacute / chronic: PS92 8/4/2/12, PS93 8/4/3/12, PS94 8/4/5/8, PS95 8/1/7/11.")

# section-7 maps
W7 = {
    "precue_contact": "2 s ending at the cue, free of DAQ licks (the window slides back to the latest lick-free gap, else the trial is dropped); " + WORK,
    "precue_LP": ("1 s ending at the cue, free of video (LP) licks &mdash; the LP default (<code>lp_licks.LP_ENL_WINDOW_S</code>; a 2 s "
                  "LP-lick-free window kept only ~48% of some animals&rsquo; trials); " + WORK),
    "enl1s_contact": "1 s ending at the cue, free of DAQ licks (the contact arm matched to the 1 s LP window); " + WORK,
    "cue": "0&ndash;2 s after the cue; " + WORK,
    "lick": "0&ndash;2 s from the first lick (first lick within 3.5 s of the cue); lick trials only",
}
R7 = {
    "REST": ("REST = time-local rest baseline (docked, not running, not licking, engaged rest frames; median in 12 bins "
             "across the session, interpolated to every frame, subtracted before trial averaging)"),
    "PRECUE": "PRECUE = each trial minus its own mean over the 1 s before the cue (the cue-evoked increment)",
}
G7 = {
    "contact": "contact = licks from the DAQ spout-contact sensor",
    "LP": ("LP = Lightning Pose video licks (hit or incomplete attempt with the first LP lick within 3.5 s; the lick "
           "window starts at that LP lick)"),
}
M7 = ("<b>No decoder.</b> <b>Measured:</b> trial-averaged widefield response per spout position (haemodynamic-corrected "
      "signal through the atlas-registered spatial basis; no model, no z-score), per session, &ge; 20 trials per position "
      "per session. <b>Window:</b> {w}. <b>Reference:</b> {r}. <b>Gate:</b> {g}. ")
POOL7 = ("<b>Pooling:</b> each animal&rsquo;s session mean divided by its own pre-stroke RMS (per position), then averaged "
         "over animals present in both epochs; difference columns = epoch &minus; pre on that scale; ratio = RMS(epoch) / "
         "RMS(pre) in the brain mask; one colour scale for all rows (the figure&rsquo;s own subtitle &lsquo;PER ROW&rsquo; is "
         "stale). <b>Stats:</b> nested animals &rarr; sessions bootstrap (2,000) on 2,022 bins (8&times; downsampled, eroded "
         "mask without bulbs / glue), studentised, family-wise &alpha; = .05 by max |z| (contours); a panel is suppressed when "
         "its significant bins are &gt; 2&times; concentrated in the rim; panel titles give animals, sessions, trials and "
         "split-half r. Pre = all curated pre-stroke sessions (June + August). " + EPO + ". ")
ANI7 = ("<b>Per animal:</b> far contra only; each animal&rsquo;s raw epoch means (not normalised), difference = epoch "
        "&minus; pre; colour scale per animal; no statistics. " + EPO + ". ")
DRIFT7 = ("<b>Drift:</b> the same per-session maps and test as the pooled map, with pre split at 08-01: June pre "
          "(12 sessions) | August pre (32) | Aug &minus; June (no stroke, scaled by each animal&rsquo;s June RMS) | acute "
          "&minus; Aug pre | chronic &minus; Aug pre (scaled by August RMS); ratio = RMS vs the reference epoch; subacute not "
          "shown; June includes learning, so the drift column is an upper bound on drift alone. ")
SC7 = {"map": "<code>wfield_local/epoch_maps.py</code> (maps <code>position_reference_maps.py</code>, test <code>beta_maps.py</code>)",
       "ani": "<code>wfield_local/epoch_maps.py</code> (per-animal family 15rpa)",
       "drift": "<code>scripts/map_drift_prepre.py</code>"}

# ---------------------------------------------------------------- per figure: title, methods, script
T, M, SC = {}, {}, {}

T["fig0-1"] = "Behaviour (engaged hit rate, licks per trial) over days since lesion: the far-contra deficit defines the epochs; PS92 / PS93 recover fastest, PS94 slowest"
M["fig0-1"] = ("<b>No model.</b> <b>Measured:</b> hit rate = share of engaged trials with a DAQ spout-contact lick within 3.5 s "
               "of the cue, per spout position (top); licks per engaged trial (bottom). Pre = one point per animal, hits and "
               "trials summed over all pre-stroke sessions. 4 animals, 120 sessions. No statistics. " + EPO + ".")
SC["fig0-1"] = "wfield_local/epoch_behaviour.py"

T["fig1-1"] = "Frozen pre-stroke spout decoder, pre-cue 2 s window: position decodable before the cue, but far contra low even at baseline"
M["fig1-1"] = DEC_F + " <b>Window:</b> 2 s ending at the cue (4 &times; 0.5 s bins), free of DAQ licks; no per-trial baseline. <b>Trials:</b> " + WORK + "; contact gate. Movement not regressed out. " + ST_ACC
SC["fig1-1"] = "wfield_local/epoch_accuracy.py"

T["fig1-2"] = "Within-session Allen-area decoder, pre-stroke sessions: 70–80% of the above-chance pre-cue code survives removing the ≥ 300 s band"
M["fig1-2"] = ("<b>Decoder:</b> REFIT within each session (logistic, C = 0.5, 5-fold GroupKFold by position block) on "
               "Allen-area features &mdash; not the frozen decoder. <b>Measured:</b> accuracy, chance 1/6, for the adopted "
               "corrected signal, the signal with the &ge; 300 s band removed (global cosine projection), and the slow band "
               "alone (&ge; 300 / &ge; 600 s). <b>Window:</b> pre-cue and post-cue 2 s. <b>Data:</b> pre-stroke sessions only "
               "(A&ndash;C); D = all 46 sessions, slow-band accuracy against median position-block duration (r = 0.75). Bars = "
               "mean of animals; no test.")
SC["fig1-2"] = "scripts/slowband_caveat_figure.py (decoding scripts/slowband_decode_check.py)"

T["fig2-1"] = "Frozen pre-stroke spout decoder, 1 s pre-cue window: far-contra accuracy roughly halves acutely in 4 / 4 animals"
M["fig2-1"] = DEC_F + " <b>Window:</b> 1 s ending at the cue, free of DAQ licks (<code>WIDEFIELD_ENL_WINDOW_S=1.0</code>). <b>Trials:</b> " + WORK + "; contact gate. " + ST_ACC
SC["fig2-1"] = "wfield_local/epoch_accuracy.py (arm grant_figures/DAQ_ENL1s)"

T["fig2-2"] = "Frozen pre-stroke spout decoder, cue window: far-contra accuracy collapses acutely and recovers only partly"
M["fig2-2"] = DEC_F + " <b>Window:</b> 0&ndash;2 s after the cue (4 bins). <b>Trials:</b> " + WORK + "; contact gate. " + ST_ACC
SC["fig2-2"] = "wfield_local/epoch_accuracy.py"

T["fig2-3"] = "Frozen pre-stroke spout decoder, lick window, LP lick gate: acute far contra falls to ~0.3, which contact gating hides"
M["fig2-3"] = (DEC_F + " <b>Window:</b> 0&ndash;2 s from the first video (LP) lick (8 &times; 0.25 s bins). <b>Trials:</b> "
               "LP attempt (hit or incomplete) with the first LP lick within 3.5 s; no-attempt and jaw-only trials excluded; "
               "sessions without cam1 use cam4 only. " + ST_ACC)
SC["fig2-3"] = "wfield_local/epoch_accuracy.py (WIDEFIELD_LICK_SOURCE=lp, grant_figures/LP_licks)"

T["fig3-1"] = "Frozen position vs behavioural-state decoders, cue window: position loses ~half its above-chance accuracy acutely, state ~10%"
M["fig3-1"] = ("<b>Decoders:</b> two FROZEN pre-stroke decoders on the same joint-LocaNMF basis. Position = the Fig 1.1 decoder, "
               "cue 0&ndash;2 s, " + WORK + ". State = logistic (C = 1.0) classifying 1 s segments (4 &times; 0.25 s bins) as "
               "rest / running / licking, licking segments anchored at the first post-cue lick. Both pre columns "
               "leave-one-session-out. <b>Measured:</b> balanced accuracy as (acc &minus; chance) / (1 &minus; chance), chance "
               "1/6 (position) and 1/3 (state). PS92 08-12 excluded from the state arm. <b>Stats:</b> none &mdash; one pooled "
               "number per bar. " + EPO + ".")
SC["fig3-1"] = "wfield_local/epoch_state.py (state decoder wfield_local/locomotor_decoder.py)"

T["fig3-2"] = "Frozen behavioural-state decoder, change from pre: a small fall, largest subacute / chronic and driven by PS94"
M["fig3-2"] = ("<b>Decoder:</b> the FROZEN state decoder of Fig 3.1. <b>Measured:</b> balanced accuracy minus pre (raw). "
               "<b>Stats:</b> two-level bootstrap (animals &rarr; sessions, 2,000); thick bar = 95% CI, thin = Bonferroni over "
               "3 epochs; * uncorrected, ** Bonferroni; no mark with &lt; 2 animals. 4 animals. " + EPO + ".")
SC["fig3-2"] = "wfield_local/epoch_state.py"

T["fig3-3"] = "Per animal, frozen encoder and frozen state decoder, cue window: template amplitude and captured shape fall in all four; state falls only in PS94"
M["fig3-3"] = ("<b>Models:</b> FROZEN pre-stroke encoder (the pre-stroke mean pattern per spout, Fig 6.1) and the FROZEN "
               "state decoder (Fig 3.1). <b>Measured:</b> left = amplitude a (Fig 6.1); middle = size-matched shape EV / "
               "within-session split-half ceiling (Fig 6.2); right = state accuracy above chance. Cue 0&ndash;2 s, " + WORK +
               "; pre = leave-one-session-out; PS95 acute = 1 session. Session means per animal, black = unweighted mean of "
               "animals; no statistics.")
SC["fig3-3"] = "per_animal_fig.py (session script, not in the repo) from the epoch_11amp / epoch_11c / epoch_13 session tables"

T["fig4-1"] = "Size-matched frozen vs refit decoders, cue window, per animal: 3 / 4 animals end displaced (F and G > 0); only PS92 returns to its origin"
M["fig4-1"] = ("<b>Decoders:</b> FROZEN pre-stroke logistic decoder fit on a random subset of whole pre-stroke position "
               "blocks, size-matched to the refit&rsquo;s training set, vs a decoder REFIT within each session (5-fold "
               "GroupKFold by block), both scored on the same trials. <b>Measured:</b> F = pre_frozen &minus; frozen "
               "(frozen-readout deficit); G = (refit &minus; frozen) &minus; (pre_refit &minus; pre_frozen) (reorganisation); "
               "both relative to the animal&rsquo;s own pre. <b>Window:</b> cue 0&ndash;2 s, " + WORK + "; 6 positions pooled. "
               "Post-stroke sessions coloured by day; no statistics.")
SC["fig4-1"] = "wfield_local/recovery_trajectory.py"

T["fig4-2"] = "Size-matched refit − frozen gap, cue window, change from pre: the chronic gap stays positive (information present but unreadable by the pre-stroke decoder)"
M["fig4-2"] = ("<b>Decoders:</b> the size-matched FROZEN vs within-session REFIT pair of Fig 4.1. <b>Measured:</b> (refit "
               "&minus; frozen) at each epoch minus the pre gap, per position. <b>Window:</b> cue 0&ndash;2 s, " + WORK + ". " + ST_ACC)
SC["fig4-2"] = "wfield_local/epoch_accuracy.py (matched frozen-vs-refit family)"

T["fig4-3"] = "No-stroke June → August drift, cue window: two months without a stroke move distance and frozen accuracy about as much as chronic stroke"
M["fig4-3"] = ("<b>Measures:</b> (top) crossnobis own-position distance of each session&rsquo;s pattern from a pooled "
               "template, in units of the animal&rsquo;s pre-stroke between-position distance; (bottom) a FROZEN logistic "
               "decoder fit on June pre (August pre for post-stroke sessions), size-matched, vs a within-session REFIT "
               "(block cross-validated). <b>Window:</b> cue 0&ndash;2 s, engaged (licking) trials only. Per animal; no "
               "statistics. Drift alone gives F 0.13&ndash;0.28 and G 0.09&ndash;0.18 (<code>summary_by_animal.csv</code>).")
SC["fig4-3"] = "scripts/june_august_drift.py"

T["fig5-1"] = "Frozen pre-stroke spout decoder by attempt class, acute, pre-cue / cue / lick windows: failed far-contra attempts keep “far” but lose the side"
M["fig5-1"] = ("<b>Decoder:</b> FROZEN pre-stroke 6-way multinomial logistic on joint-LocaNMF components, fit on pre-stroke "
               "engaged trials (pre rows leave-one-session-out). <b>Measured:</b> P(true spout) (top, chance 1/6) and share "
               "decoded as any far spout (bottom). <b>Windows:</b> pre-cue 1 s (LP-lick-free), cue 0&ndash;2 s, lick 0&ndash;2 s "
               "from the first video lick. <b>Classes</b> from video attempts (LP): pre-stroke hit, hit, incomplete, no "
               "attempt; far-contra spout, acute epoch; classes with &ge; 5 trials; 4 animals. Bars = mean &plusmn; SEM over "
               "trials; no test.")
SC["fig5-1"] = "fig_goal_summary.py (encoding window&rsquo;s page script) from wfield_local/attempt_coding.py"

T["fig5-2"] = "Frozen pre-stroke spout decoder by attempt class across recovery, cue and lick windows: failed attempts stay low, hits recover but stay below pre"
M["fig5-2"] = ("<b>Decoder:</b> as Fig 5.1 (FROZEN). <b>Measured:</b> far-contra P(true spout) per attempt class and epoch; "
               "solid line = pre-stroke hits, dotted = chance 1/6. <b>Windows:</b> cue 0&ndash;2 s and lick 0&ndash;2 s from "
               "the first video lick. Classes with &ge; 5 trials; trial counts under each axis. Mean &plusmn; SEM over "
               "trials; no test.")
SC["fig5-2"] = "fig_goal_epochs.py (encoding window&rsquo;s page script) from wfield_local/attempt_coding.py"

ENC = ("<b>Encoder:</b> FROZEN pre-stroke one-hot position encoder, i.e. each spout&rsquo;s pre-stroke mean pattern on "
       "joint-LocaNMF components (session z-scored on engaged trials); pre sessions scored leave-one-session-out, post "
       "sessions against the full pre pool. <b>Window:</b> cue 0&ndash;2 s; post = licking trials + misses while working, "
       "pre = licking trials; positions with &ge; 10 trials; no movement regressors. ")
T["fig6-1"] = "Frozen pre-stroke encoder, cue window: template amplitude falls to ~0.36× acutely and is back near pre by chronic"
M["fig6-1"] = ENC + ("<b>Measured:</b> amplitude a = &Sigma;m&middot;p / &Sigma;p&middot;p after centring each pattern on the "
                     "session&rsquo;s across-position mean (1 = no change). Values: pre 0.94, acute 0.36**, subacute 0.67**, "
                     "chronic 0.88 (n.s.). " + ST_EPO + EPO + ".")
SC["fig6-1"] = "wfield_local/epoch_matching.py (terms wfield_local/grant_encoder.py)"

T["fig6-2"] = "Frozen pre-stroke encoder vs within-session ceiling, cue window: the pre-stroke template captures less shape even when the session has more to capture"
M["fig6-2"] = ENC + ("<b>Measured:</b> EV after one best rescale (shape only), all on the same half-session: ceiling = scored "
                     "against the other half of the same session; frozen (matched) = against a same-size random draw from "
                     "the pre pool; frozen (all pre) = against the whole pool. Ceiling 0.71 &rarr; 0.57 acute &rarr; 0.78* "
                     "chronic; matched 0.43 &rarr; 0.10** &rarr; 0.29**. " + ST_EPO + EPO + ".")
SC["fig6-2"] = "wfield_local/epoch_matching.py"

T["fig6-3"] = "Frozen pre-stroke encoder, cue window, per spout: the captured share of far-contra shape collapses acutely and stays below pre"
M["fig6-3"] = ENC + ("<b>Measured:</b> per session and position, matched frozen EV / ceiling, clipped to [0, 1]; cells with "
                     "ceiling &lt; 0.10 dropped. " + ST_EPO + EPO + ".")
SC["fig6-3"] = "wfield_local/epoch_matching.py"

ATT = ("<b>No model.</b> <b>Measured:</b> trial-averaged pixel maps, rest-referenced (time-local rest baseline), through "
       "the atlas-registered basis at half resolution. <b>Window:</b> 0&ndash;2 s from the first video (LP) lick. "
       "<b>Classes</b> from video attempts: in the lick window only hit and incomplete exist (no-lick classes blank). A class "
       "needs &ge; 5 trials in a session, then a session-weighted mean (pre-stroke incomplete and pooled incomplete are "
       "trial-weighted, no floor); bottom row = mean of animal means (&ge; 2 animals); per-row symmetric 99th-percentile "
       "colour scale. No statistics. ")
T["fig8-1"] = "Rest-referenced lick-window maps, far contra, by attempt class and epoch: failed attempts look like weak hits or like misses"
M["fig8-1"] = ATT + EPO + "."
SC["fig8-1"] = "wfield_local/attempt_maps.py"
T["fig8-2"] = "The same far-contra attempt-class maps minus each animal’s pre-stroke hits"
M["fig8-2"] = ATT + "<b>Reference:</b> minus that animal&rsquo;s pre-stroke hits."
SC["fig8-2"] = "wfield_local/attempt_maps.py"

T["fig9-1"] = "Shared mode vs its trial-matched pre-stroke ceiling, all windows, per animal: it rotates (overlap below ceiling) and grows (magnitude above)"
M["fig9-1"] = ("<b>Measure:</b> the shared (condition-independent) mode = leading components (to 90% variance, &le; 8; K = 2&ndash;3) "
               "of each epoch&rsquo;s grand-mean trajectory (baseline [&minus;3, &minus;2) s) on joint-LocaNMF components, cortex only. "
               "Overlap = mean squared cosine of the principal angles to the pre mode; magnitude = Frobenius norm of the grand-mean "
               "deviation / pre; row C = unremoved amplitude. Each divided by its ceiling = median of 200 trial-matched splits "
               "of pre vs pre (1 = no change). Columns = pre-cue, cue, lick; success trials. Per animal; no test (the ceiling "
               "is not a p-value); only PS94 has a chronic point. " + EPO + ".")
SC["fig9-1"] = "scripts/cd_geometry_figure.py (wfield_local/cd_trajectories.py, wfield_local/cd_overlap_null.py)"

T["fig9-2"] = "Cross-position projections ΔM, pre-cue window: changes concentrate chronically; no robust usurpation (far contra → far middle in 2 / 4 animals)"
M["fig9-2"] = (CD + " <b>Measured:</b> full 6 &times; 6 matrix M[i, j] = trials at position i projected onto the frozen "
               "direction of position j; top = level, bottom = &Delta;M = epoch &minus; pre within animal, mean over 4 animals. "
               "<b>Window:</b> pre-cue [&minus;2, 0) s. <b>Stats:</b> within-animal permutation of the session epoch labels "
               "(2,000); FILLED dot = family-wise significant (max statistic over the 36 cells, p &lt; .05), hollow = per cell "
               "only; small number = animals sharing the sign. Box = USURPED (M[i, j] &gt; M[j, j] while j&rsquo;s own diagonal "
               "keeps &ge; 50% of pre); dashed box = &lt; 3 / 4 animals (k / 4); hatched = VACATED. Family-wise cells "
               "acute / subacute / chronic: 1 / 0 / 5.")
SC["fig9-2"] = "scripts/cd_migration.py"

for k, _w in (("fig9-3", "pre-cue"), ("fig9-4", "cue")):
    M[k] = (CD + CD_LOO + " <b>Measured:</b> each position&rsquo;s projection on its own direction, window mean, raw pole "
            "units; grey band = range of the held-out pre sessions, dashed = their median; rows = animals, x = days from "
            "stroke. No statistic. " + EPO + ".")
    SC[k] = "scripts/cd_loo_figures.py (scored by scripts/cd_loo_sessions.py)"
T["fig9-3"] = "Own-position coding direction, leave-one-out, pre-cue window, per animal: far contra falls acutely in 3 / 4 animals"
T["fig9-4"] = "Own-position coding direction, leave-one-out, cue window, per animal: far contra drops pre → acute in 3 / 4 (not PS95)"
T["fig9-5"] = "Own-position coding-direction trajectories, leave-one-out, cue-aligned, per animal: the acute far-contra trace sits below held-out pre"
M["fig9-5"] = (CD + CD_LOO + " <b>Measured:</b> own-position projection over &minus;3 to +4 s from the cue (0.2 s smoothing), "
               "mean &plusmn; SEM over sessions per epoch; shaded = readout window. No statistic; do not read onset timing "
               "(haemodynamic signal).")
SC["fig9-5"] = "scripts/cd_loo_figures.py"
for k, _w, t in (("fig9-6", "pre-cue", "Own-position coding direction minus held-out pre, pre-cue window, pooled: far contra below pre at every post epoch (subacute 4 / 4*)"),
                ("fig9-7", "cue", "Own-position coding direction minus held-out pre, cue window, pooled: far contra subacute −.15*, 4 / 4"),
                ("fig9-8", "lick", "Own-position coding direction minus held-out pre, lick window, pooled: no consistent far-contra change")):
    T[k] = t
    M[k] = (CD + CD_LOO + " <b>Measured:</b> each animal&rsquo;s value minus its own mean held-out pre (same position and "
            "window), pole units, pre = 0; thin lines = animals, black = mean &plusmn; SEM. <b>Stats:</b> paired nested "
            "bootstrap (animals &rarr; sessions, same animal draw for pre and post, 4,000); * = 95% CI excludes 0, "
            "<b>uncorrected</b> (6 positions &times; 3 epochs &times; 3 windows); k / 4 = animals below their pre.")
    SC[k] = "scripts/cd_loo_pooled.py (stats cd_loo_pooled_stats.csv)"
for k, w in (("fig9-9", "pre-cue"), ("fig9-10", "cue"), ("fig9-11", "lick")):
    T[k] = f"Own-position coding-direction trajectories minus held-out pre, {w}-aligned, pooled over animals"
    M[k] = (CD + CD_LOO + " <b>Measured:</b> each animal&rsquo;s session-mean trace minus its pre window-mean scalar (so pre "
            "is ~0 only inside the window), then mean &plusmn; SEM over animals; epochs with &lt; 2 animals skipped; "
            "shaded = readout window. No statistic; do not read onset timing.")
    SC[k] = "scripts/cd_loo_pooled.py"
T["fig9-9"] += ": the far-contra deficit holds through the pre-cue window"

T["fig10-1"] = "Crossnobis distance to the pre-stroke pattern, cue window: the acute change is focal at far contra, the chronic change broad"
M["fig10-1"] = ("<b>Measure:</b> cross-validated crossnobis distance between each session&rsquo;s per-position pattern and "
                "the pre-stroke pool, on joint-LocaNMF components (pre scored leave-one-session-out), in units of the "
                "animal&rsquo;s mean pre between-position distance; plotted = own-position (diagonal) distance, epoch &minus; "
                "pre; not gain-invariant. <b>Window:</b> cue 0&ndash;2 s, " + WORK + ". Acute far contra +1.02; chronic every "
                "position +0.36 to +0.84. " + ST_ACC.replace("bars = trial-pooled accuracy, dots = sessions; ", "") )
SC["fig10-1"] = "wfield_local/epoch_accuracy.py (distances wfield_local/grant_geometry.py)"

T["fig10-2"] = "Session-to-session crossnobis change rate, cue window, per animal: chronic change is no faster than pre-stroke drift in 4 / 4"
M["fig10-2"] = ("<b>Measure:</b> pairwise own-position crossnobis distance between sessions (mean of 15 half-splits, Fig 10.1 "
                "units) against the days between them; slope of pre&ndash;pre pairs = drift floor, slope of chronic&ndash;chronic "
                "pairs = chronic rate. <b>Window:</b> cue 0&ndash;2 s, " + WORK + ". Sessions per animal: 11 pre; chronic "
                "12 / 12 / 8 / 11. <b>Stats:</b> session bootstrap (2,000) of each slope; 95% CI of chronic &minus; pre slope "
                "from index-paired draws; all 4 CIs include 0.")
SC["fig10-2"] = "scripts/chronic_drift_vs_reorg.py"

T["fig11-1"] = "Success-trained pre-cue decoder, pre-stroke, by state: the position code persists on stopped trials in 2 / 4 animals, not distinguishable from engaged misses"
M["fig11-1"] = ("<b>Decoder:</b> one decoder per animal trained on success trials, joint-LocaNMF components, position blocks "
                "held out (GroupKFold), applied to success, miss-while-working and stopped trials (stopped = after the "
                "session&rsquo;s terminal engagement collapse). <b>Window:</b> 2 s ending at the cue, strict lick-free. "
                "<b>Data:</b> pre-stroke only, 11 sessions per animal, 4 animals (PS92 3 / 6 positions). <b>Stats:</b> each arm "
                "against its own permutation null (black tick); paired block-bootstrap 95% CIs; hatched = at its null. "
                "Stopped above null in 2 / 4 (PS94 0.25, PS95 0.30); stopped / miss 0.58 and 0.86, CIs include 1.")
SC["fig11-1"] = "scripts/enl_decode_figure.py (decoding wfield_local/enl_decode.py)"

T["fig11-2"] = "Within-session pre-cue decoder with movement removed, across recovery: the residual position code tracks the raw code"
M["fig11-2"] = ("<b>Decoder:</b> REFIT within each session (logistic, C = 0.5, 4 &times; 0.5 s bins, GroupKFold by block) on "
                "the session&rsquo;s LocaNMF components. Arms: raw; residual = ridge on running + 4-camera video motion energy "
                "regressed out within each fold; null_3 = top-3 movement-potent dimensions projected out. <b>Window:</b> "
                "[cue &minus; 2 s, cue], lick-free, engaged trials. Chance 1/6. Curated sessions with &ge; 60 trials, by day, "
                "per animal; no statistics plotted (label-shuffle p in the CSV).")
SC["fig11-2"] = "scripts/enl_movement_multisession.py (plotted by a session script, not in the repo)"

T["fig12-1"] = "Frozen pre-stroke rest decoder, duration- and lick-gap-matched: the rest position code falls to ~30% acutely and ~half by chronic"
M["fig12-1"] = ("<b>Decoder:</b> FROZEN pre-stroke logistic (C = 0.1) on joint-LocaNMF rest-period features (4 bins per rest "
                "period); pre leave-one-session-out; no refit arm. <b>Rest</b> = docked spout, between two working trials at "
                "the same position, not running, lick buffer [0.5, 1.0] s; train and test restricted to the central 80% of "
                "the pre-stroke duration and lick-gap ranges. <b>Measured:</b> balanced accuracy (chance 1/6); retained = "
                "(acc &minus; 1/6) / (5/6) relative to the animal&rsquo;s own pre; task (0.496) and state (0.902) lines are quoted "
                "values. Sessions pre 44 / acute 16 / subacute 17 / chronic 43. <b>Null:</b> block label permutation (50) "
                "&mdash; the legend&rsquo;s &lsquo;circular-shift&rsquo; is a label error; no CIs. Retained 0.28 acute, 0.54 "
                "chronic; PS95 below chance acutely.")
SC["fig12-1"] = "scripts/rest_migration/rest_frozen_decoder.py (--match-duration --match-lickgap, restdock05)"

DEC_M = ("<b>Decoders:</b> size-matched FROZEN pre-stroke logistic vs within-session REFIT (5-fold block), same trials, "
         "cue 0&ndash;2 s, " + WORK + " (Fig 4.1). Post-stroke sessions only, n = 76, 4 animals; pre = pooled pre-stroke "
         "sessions. ")
T["fig13-1"] = "Frozen (matched) and refit decoders, cue window, mixed model on log-days: the frozen decoder recovers while reorganisation G grows"
M["fig13-1"] = DEC_M + LMM + " y ~ lday, lday = ln(days since stroke), 0 at day 1. Frozen 0.41 at day 1, +0.064 per e-fold (z 3.9); G +0.030 per e-fold (z 4.5, q 2e-5)."
SC["fig13-1"] = "scripts/imaging_lmm.py (data wfield_local/recovery_trajectory.py)"
T["fig13-2"] = "Frozen and refit decoders, cue window, mixed model on the behavioural deficit: a frozen deficit and G remain at full behavioural recovery"
M["fig13-2"] = (DEC_M + LMM + " y ~ deficit, deficit = (session &minus; pre) / pre of far-contra time to contact (August pre "
                "mean; 0 = full recovery). At full recovery: frozen 0.63, F 0.086 (z 2.4, q 0.019), G 0.10 (z 4.2); refit "
                "deficit ~0.")
SC["fig13-2"] = "scripts/imaging_lmm.py"
T["fig13-3"] = "Frozen pre-stroke encoder EV, cue and lick windows, mixed model on log-days: a large day-1 drop, then partial log-time recovery"
M["fig13-3"] = ("<b>Encoder:</b> the FROZEN pre-stroke encoder of Fig 6.1; panels = frozen EV and EV after rescale, cue 0&ndash;2 s "
                "(working trials) and lick 0&ndash;2 s from the first lick (lick trials); pre scored leave-one-session-out, August "
                "pre only. " + LMM + " y ~ post + post:lday; n = 108 sessions. Cue frozen EV step &minus;0.97 (z &minus;10.2), "
                "then +0.21 per e-fold (z 7.2); after rescale &minus;0.49, +0.074 per e-fold.")
SC["fig13-3"] = "scripts/imaging_lmm.py (data scripts/epoch_sessions_long.py)"
TREND = ("<b>Measures</b> (within-session, refit): ceiling = cortex lick-window encoder R&sup2;(category) &minus; R&sup2;(none) "
         "from the no-kinematics encoder run; spread = RMS over the six spouts of the response-deviation maps, cue 0.5&ndash;1.5 s; "
         "spread ENL = the same in the pre-cue window. August-onward sessions, n = 107, 4 animals. ")
T["fig13-4"] = "Within-session position coding (encoder ceiling, per-spout spread), mixed model on log-days: rises steadily after stroke, no acute step"
M["fig13-4"] = TREND + LMM + " y ~ post + post:lday."
SC["fig13-4"] = "scripts/imaging_lmm.py (data scripts/position_coding_trend_recovery.py)"
T["fig13-5"] = "Position-coding ceiling and spread, mixed model on the time-to-contact deficit: spread scales with recovery, the ceiling does not"
M["fig13-5"] = TREND + LMM + " y ~ post + post:deficit, deficit = far-contra time to contact (as Fig 13.2)."
SC["fig13-5"] = "scripts/imaging_lmm.py"
T["fig13-6"] = "Position-coding ceiling and spread, mixed model on the contact deficit: both scale inversely with the deficit and exceed pre at full recovery"
M["fig13-6"] = (TREND + LMM + " y ~ post + post:deficit, deficit = (pre &minus; session) / pre of the far-contra contact "
                "fraction (engaged trials, licks 0&ndash;3.5 s after the cue). At full recovery ceiling +0.036 (z 4.4), spread "
                "+0.0016 (z 4.7), spread ENL +0.0013 (z 2.3).")
SC["fig13-6"] = "scripts/imaging_lmm.py --deficit contact"

# section 7: (fid, kind, window, ref, gate, title)
S7 = [
    ("fig7-1", "map", "precue", "REST", "contact", "Pre-cue 2 s · REST · contact response maps, pooled: the anticipatory position signal before the cue"),
    ("fig7-2", "drift", "precue", "REST", "contact", "Pre-cue 2 s · REST · contact, June → August drift: chronic far-contra fall is drift-sized; chronic far-middle loss (.53) beyond drift"),
    ("fig7-3", "map", "precue", "REST", "LP", "Pre-cue 1 s · REST · LP-lick response maps, pooled: the anticipatory signal on video-lick-free trials"),
    ("fig7-4", "drift", "precue", "REST", "LP", "Pre-cue 1 s · REST · LP-lick, June → August drift: far contra as contact; chronic far-middle loss (.30) far beyond drift"),
    ("fig7-5", "map", "enl1s", "REST", "contact", "Pre-cue (ENL) 1 s · REST · contact response maps, pooled: no acute far-contra loss before the cue (1.67)"),
    ("fig7-6", "drift", "enl1s", "REST", "contact", "Pre-cue (ENL) 1 s · REST · contact, June → August drift: chronic far-contra fall drift-sized; far-middle chronic (.35) beyond drift"),
    ("fig7-7", "map", "cue", "REST", "contact", "Cue · REST · contact response maps, pooled: acute far contra .48 of pre, far middle .66, near ipsi up to 1.43"),
    ("fig7-8", "ani", "cue", "REST", "contact", "Cue · REST · contact far-contra maps, per animal: the acute loss is seen animal by animal"),
    ("fig7-9", "drift", "cue", "REST", "contact", "Cue · REST · contact, June → August drift: far-contra acute loss (.49) beyond drift (.95); far middle not separable from drift"),
    ("fig7-10", "map", "cue", "REST", "LP", "Cue · REST · LP-lick response maps, pooled: the same as the contact gate"),
    ("fig7-11", "drift", "cue", "REST", "LP", "Cue · REST · LP-lick, June → August drift: identical to contact; the gate does not matter in the cue window"),
    ("fig7-12", "map", "cue", "PRECUE", "contact", "Cue · PRECUE · contact response maps, pooled: the cue-evoked far-contra increment falls to .18 acutely"),
    ("fig7-13", "drift", "cue", "PRECUE", "contact", "Cue · PRECUE · contact, June → August drift: far-contra (.18) and far-middle (.55) acute losses both beyond drift"),
    ("fig7-14", "map", "cue", "PRECUE", "LP", "Cue · PRECUE · LP-lick response maps, pooled: the same as the contact gate"),
    ("fig7-15", "drift", "cue", "PRECUE", "LP", "Cue · PRECUE · LP-lick, June → August drift: far contra .18 vs drift 1.11; the same as contact"),
    ("fig7-16", "map", "lick", "REST", "contact", "Lick · REST · contact response maps, pooled: the acute far-contra panel is empty (< 20 contact trials)"),
    ("fig7-17", "drift", "lick", "REST", "contact", "Lick · REST · contact, June → August drift: no acute far-contra estimate; near-spout acute changes rise, opposite to drift"),
    ("fig7-18", "map", "lick", "REST", "LP", "Lick · REST · LP-lick response maps, pooled (headline): acute far contra .67 of pre, back to ~1.0 by subacute"),
    ("fig7-19", "ani", "lick", "REST", "LP", "Lick · REST · LP-lick far-contra maps, per animal: acute −55 / −48 / −38 / +2 % (PS95 one session)"),
    ("fig7-20", "drift", "lick", "REST", "LP", "Lick · REST · LP-lick, June → August drift: far-contra acute loss (.67) beyond drift (1.04); no far-middle loss"),
    ("fig7-21", "map", "lick", "PRECUE", "contact", "Lick · PRECUE · contact response maps, pooled: cue-referenced lick maps; acute far contra empty under contact"),
    ("fig7-22", "drift", "lick", "PRECUE", "contact", "Lick · PRECUE · contact, June → August drift: the far-contra increment grew June → August (1.32, learning); no acute estimate"),
    ("fig7-23", "map", "lick", "PRECUE", "LP", "Lick · PRECUE · LP-lick response maps, pooled: the LP twin of Fig 7.21, with an acute far-contra estimate"),
    ("fig7-24", "drift", "lick", "PRECUE", "LP", "Lick · PRECUE · LP-lick, June → August drift: far contra drift 1.24 up vs acute .56 down, beyond drift"),
]
for fid, kind, w, r, g, t in S7:
    wk = w + "_" + g if w in ("precue", "enl1s") else w
    T[fid] = t
    M[fid] = M7.format(w=W7[wk], r=R7[r], g=G7[g]) + {"map": POOL7, "ani": ANI7, "drift": DRIFT7 + POOL7.split("<b>Stats:</b>")[1].split("Pre = all")[0].join(["<b>Stats:</b>", ""])}[kind]
    SC[fid] = SC7[kind]

# ---------------------------------------------------------------- caption corrections from the trace
rep("Each point = one session; shaded = acute epoch, dotted = first subacute day.",
    "Each point = one session; shaded = acute epoch (subacute starts at its right edge), dashed = first chronic day. The "
    "figure&rsquo;s own note &lsquo;only PS92 has stabilised&rsquo; is stale: all four animals now have a chronic day.")
rep("Bars = pooled session means, dots = sessions coloured by animal.", "Bars = trial-pooled accuracy, dots = sessions coloured by animal.")
rep("Bars = pooled epoch means, dots = sessions by animal, chance dotted.", "Bars = trial-pooled accuracy per epoch, dots = sessions by animal, chance dotted.")
rep("<b class=\"h\">Fig 1.2 &middot; slow-band sensitivity.</b> Each epoch with and without the &ge; 300 s band. Read the drop as how much of the pre-cue number is slow drift (0.03&ndash;0.09).",
    "<b class=\"h\">Fig 1.2 &middot; slow-band sensitivity (pre-stroke sessions).</b> Pre-cue accuracy 0.41 with the adopted "
    "signal, 0.35 without the &ge; 300 s band: the drop (0.03&ndash;0.09 per animal) is the part of the pre-cue number that is "
    "slow drift; it is largest where position blocks are longest (panel D).")
rep("<b class=\"h\">Fig 4.1 &middot; frozen vs refit over recovery, per animal.</b> The vertical gap between the two lines is the code that a pre-stroke reader can no longer use.",
    "<b class=\"h\">Fig 4.1 &middot; frozen-readout deficit F against reorganisation G, per animal.</b> Each point = one post-stroke "
    "session (colour = days from lesion; square = first, star = last); grey = pre-stroke sessions, cross = their mean &plusmn; SEM, "
    "i.e. the origin is the animal&rsquo;s own pre. Height above 0 = code that moved (a refit decoder finds it, the frozen one "
    "cannot); distance below the dashed diagonal F = G = code no decoder recovers. Final F / G: PS92 &minus;.06 / .00, PS93 "
    ".15 / .27, PS94 .17 / .18, PS95 .18 / .19.")
rep("Projection of far-contra trials onto each other position's direction, change from pre; boxed = family-wise significant (pre early-vs-late null: 0 cells).",
    "Every position&rsquo;s trials projected onto every position&rsquo;s frozen direction, change from pre (methods above). The "
    "one acute family-wise cell is far contra on far middle, but usurpation holds in only 2 / 4 animals.")
rep("<b class=\"h\">Fig 12.1 &middot; frozen vs refit rest decoder</b> (duration- and gap-matched).",
    "<b class=\"h\">Fig 12.1 &middot; frozen pre-stroke rest decoder</b> (duration- and lick-gap-matched).")
rep("Far contra: acute &minus;.21 [&minus;.49, +.07] 3 / 4, subacute &minus;.15 [&minus;.26, &minus;.03]* 4 / 4, chronic &minus;.13 [&minus;.32, +.11] 3 / 4.",
    "Far contra: acute &minus;.21 [&minus;.49, +.07] 3 / 4, subacute &minus;.15 [&minus;.26, &minus;.03]* 4 / 4, chronic &minus;.13 [&minus;.32, +.11] 3 / 4. "
    "Far middle acute &minus;.47* (3 / 4).")
# LP pre-cue is 1 s (lp_licks.LP_ENL_WINDOW_S), not 2 s
rep("Fig 7.3 &middot; PRE-CUE 2 s window &middot;", "Fig 7.3 &middot; PRE-CUE 1 s window (LP) &middot;")
rep("Fig 7.4 &middot; PRE-CUE 2 s window &middot;", "Fig 7.4 &middot; PRE-CUE 1 s window (LP) &middot;")
rep("The anticipatory position signal before the cue. LP twin: Fig 7.3.",
    "The anticipatory position signal before the cue. The LP-lick version (Fig 7.3) uses a 1 s window; its matched contact twin is Fig 7.5.")
rep("Contact twin: Fig 7.1.</figcaption>", "1 s window, the LP default; matched contact twin (1 s, DAQ-lick-free): Fig 7.5; the 2 s contact map is Fig 7.1.</figcaption>")
rep("because the trials are lick-free the gate barely matters, so there is no LP twin.",
    "it is the contact arm matched to the 1 s LP-lick-free window of Fig 7.3, so the two differ only in the lick definition.")
rep("<h4 class=\"h7\">PRE-CUE 2 s window, REST reference</h4>", "<h4 class=\"h7\">PRE-CUE window, REST reference (2 s contact; 1 s LP)</h4>")
rep("Windows, always in this order: PRE-CUE 2 s, ENL 1 s (lick-free pre-cue),",
    "Windows, always in this order: PRE-CUE (2 s under the contact gate, 1 s under the LP gate), ENL 1 s (DAQ-lick-free pre-cue, matched to the 1 s LP window),")
rep("alt=\"PRE-CUE 2 s window - REST reference - LP-lick gate", "alt=\"PRE-CUE 1 s window - REST reference - LP-lick gate", 2)
rep("<tr><td>PRE-CUE 2 s &middot; REST</td>", "<tr><td>PRE-CUE &middot; REST (2 s contact, 1 s LP)</td>")
rep("<td>no twin (lick-free trials)</td>", "<td><a href=\"#fig7-3\">7.3</a> is the 1 s LP match</td>")

# ---------------------------------------------------------------- insert methods + script, titles
figs = re.findall(r'<figure id="([^"]+)">', s)
assert set(figs) == set(T) == set(M) == set(SC), (set(figs) ^ set(T), set(figs) ^ set(M))


def addm(m):
    fid, body = m.group(1), m.group(0)
    h = re.search(r'<b class="h">(Fig [0-9.]+) &middot;.*?</b>', body, re.S)
    assert h, fid
    script = SC[fid]
    sc_html = script if script.startswith("<code>") else "<code>" + script.split(" (")[0] + "</code>" + (" (" + script.split(" (", 1)[1] if " (" in script else "")
    body = body.replace(h.group(0), h.group(0) + " " + M[fid] + " <b>Reading:</b>", 1)
    body = body.replace("</figcaption>", " <b>Script:</b> " + sc_html + ".</figcaption>", 1)
    title = '<div class="ftitle"><span class="fnum">%s</span>%s</div>' % (h.group(1), html.escape(T[fid], quote=False))
    return body.replace("<img", title + "<img", 1)


s = re.sub(r'<figure id="([^"]+)">.*?</figure>', addm, s, flags=re.S)
# the 'Reading:' marker is empty where the caption had nothing after the label
s = s.replace(" <b>Reading:</b> <b>Script:</b>", " <b>Script:</b>")
# CSS + sidebar uses the title
css = (".ftitle { font-weight:600; font-size:1.02rem; line-height:1.35; color:var(--fg); margin:0 0 8px; text-wrap:balance }\n"
       ".ftitle .fnum { color:var(--muted); font-weight:600; margin-right:6px }\n")
rep("</style>", css + "</style>")
rep("var txt = b ? b.textContent.replace(/[.:]$/, '') : f.id;",
    "var ft = f.querySelector('.ftitle');\n        var txt = ft ? ft.textContent : (b ? b.textContent.replace(/[.:]$/, '') : f.id);")
P.write_text(s, encoding="utf8")
print("figures", len(figs), "titles", s.count('class="ftitle"'))

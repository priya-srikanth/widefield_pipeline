"""Section 23 (Movement page) -> sync-end quiet run (a8 v2_quiet_lp_c1guard_syncend_20261118). Writes s23_page.html."""
import re

SRC = r"C:/Users/SabatiniLab/.claude/projects/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/tool-results/artifact-77c5daa3-1795016757-1e20.html"
OUT = r"C:/Users/SabatiniLab/AppData/Local/Temp/claude/179e1046909c5be2/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/s23_page.html"
s = open(SRC, encoding="utf8").read()
n0 = len(s)


def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, (c, old[:120])
    s = s.replace(old, new)


RUN = ("<code>rebuild_20261103/labcams/&lt;tag&gt;/cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118/</code> "
       "(whole chain re-extracted with the last-sync end, code 871be8f9, runner <code>quiet_c1guard/run_chain_syncend.sh</code>; "
       "old vs new numbers: <code>docs/status/quiet_restyle_20261118/QUIET_SYNCEND_OLD_NEW_20261118.md</code>)")
QF = ("<b>Quiet frames:</b> engaged part of the session &mdash; before the quit and before the last sync pulse "
      "(<code>QS_ENGAGED_END=lastsync</code>; the task controller stops 0&ndash;25 s after the last cue, so imaging after "
      "the task is not counted; Priya 11-18) &mdash; with no lick and no running bout within 1.5 s before or after (spout may be out).")
LK = ("<b>Licks:</b> DAQ spout contact OR LP video lick, whole session: cam4 <code>licks_all</code> plus cam1-only events that are "
      "non-droplet with med_out &ge; &minus;120 px (cam1-only labels far from the mouth are not counted); PS93 08-13 has no behavior "
      "video, so its quiet frames use DAQ licks only.")
TH = "<b>Threshold:</b> k = 4 &times; the animal&rsquo;s own August-pre robust SD (1.4826 &times; MAD). <b>Baseline:</b> each animal&rsquo;s own August pre."
SESS = "108 sessions (pre 32, 8 per animal; acute 16, subacute 17, chronic 43)"
STARS = ("Open markers = sessions minus their animal&rsquo;s own pre mean (shape = animal, color = epoch); filled = animal-weighted "
         "change with its 95% CI; <b>*</b> = nested animal &rarr; session bootstrap 95% CI of the change excludes 0 (primary); "
         "&dagger; = mixed model q &lt; .05 (backup, FDR within measure); k/4 = animals changing in the pooled direction (top band).")


def data_block(extra=""):
    return (f"<b>Data:</b> rebuilt base with the cam1 guard, analyzed time ending at the last sync pulse, {RUN}; {SESS}. "
            f"{QF} {LK} {TH}{extra}")


def set_caption(fid, cap):
    global s
    a = s.index(f'<figure id="f-{fid}"')
    b = s.index("<figcaption>", a) + len("<figcaption>")
    c = s.index("</figcaption>", b)
    s = s[:b] + cap + s[c:]


def set_img(old, new):
    rep(f'<img src="{old}"', f'<img src="{new}"')


def set_title(fid, old, new, short=None):
    """ftitle text (after the [licks...] tag), alt, summary line, data-short."""
    global s
    a = s.index(f'<figure id="f-{fid}"')
    e = s.index("</figure>", a)
    seg = s[a:e]
    assert seg.count(old) >= 2, (fid, seg.count(old))
    seg = seg.replace(old, new)
    if short:
        seg = re.sub(r'data-short="[^"]*"', f'data-short="{short}"', seg, count=1)
    s = s[:a] + seg + s[e:]
    d = s.rfind("<details", 0, a)
    if d != -1 and s.find("</summary>", d) < a and f'data-for="f-{fid}"' in s[d:a]:
        se = s.index("</summary>", d)
        s = s[:d] + s[d:se].replace(old, new) + s[se:]


# ---------------- section intro ----------------
rep("numbers on the rebuilt base: <code>docs/status/deck_swap_20261117/quiet_c1guard/QUIET_C1GUARD_OLD_NEW_20261117.md</code>.",
    "numbers on the rebuilt base with the analyzed time ending at the last sync pulse: "
    "<code>docs/status/quiet_restyle_20261118/QUIET_SYNCEND_OLD_NEW_20261118.md</code>.")
rep("<p class=\"take\"><b>Data and lick source:</b> Figs 23.1, 23.2, 23.4&ndash;23.6, 23.11, 23.15, 23.17, 23.18 and 23.21 are on the rebuilt base with the cam1 guard: quiet frames exclude",
    "<p class=\"take\"><b>Data and lick source:</b> Figs 23.1, 23.2, 23.4&ndash;23.6, 23.11, 23.15, 23.17, 23.18 and 23.21 are on the rebuilt base with the cam1 guard, and the analyzed time ends at the last sync pulse (engaged = before the quit and before the last sync edge, so imaging after the task controller stopped is not counted; Priya 11-18). Their stars are the nested animal &rarr; session bootstrap of the change vs each animal&rsquo;s own pre (primary), with the mixed model as a dagger backup. Quiet frames exclude")
OLD_READ_A = s.index('<p class="take"><b>Reading:</b> acutely, orofacial cortex in engaged quiet')
OLD_READ_B = s.index("</p>", OLD_READ_A) + 4
s = s[:OLD_READ_A] + (
    '<p class="take"><b>Reading:</b> acutely, orofacial cortex in engaged quiet looks like pre-stroke disengagement &mdash; a local, '
    "non-movement event type that is normally rare while engaged becomes ~7&times; more frequent (&times;7.3 face-regressed, 4/4; 23.1, "
    "23.4, 23.8), the event-map difference is mostly WHICH events occur (23.2, 23.7), and orofacial 1&ndash;4 Hz power rises with no "
    "lesion-side bias (23.3; the earlier &lsquo;ipsilesional &gt; contralesional synchronization&rsquo; was withdrawn 11-01 &mdash; it "
    "came from a non-calcium 7&ndash;11 Hz peak). The face-regressed rate stays above pre chronically in both hemispheres "
    "(ipsilesional 4/4, contralesional 3/4; 23.1). Not face movement (23.1, 23.17), not interictal (23.13, 23.19); at 415 nm "
    "(calcium-insensitive) quiet power and event rate do not rise, though the 415 response at the events rises slightly acutely, "
    "mostly in PS94 (23.11); breathing interim negative (23.12). The acute rise holds in every face-stillness bin and is largest when "
    "the face is still (23.4). The events also occur in task windows; ITI and quiet rates stay above pre chronically, pre-cue rates "
    "ipsilesionally (23.15). A pre-cue event goes with a weaker frozen-decoder position code in every epoch, pre-stroke included "
    "&mdash; it marks a state, not a stroke effect &mdash; and goes with more failed far-contra attempts only subacutely (+0.04, 4/4; "
    "23.5). All quiet measures follow the recovery time course, none beyond it (23.14).</p>") + s[OLD_READ_B:]

# ---------------- status-table row ----------------
rep("Face-regressed orofacial events &times;6.6 acute, &times;3.3 subacute, &times;1.7 chronic (rebuilt base with the cam1 guard; 4/4)",
    "Face-regressed orofacial events &times;7.3 acute, &times;3.7 subacute, &times;1.9 chronic (rebuilt base with the cam1 guard, analyzed time ending at the last sync pulse; acute 4/4)")
rep("rise holds at matched face stillness except in the most active bin; not haemodynamic, not interictal;",
    "the acute rise holds at matched face stillness in every bin; no 415 nm power or rate rise (a small 415 response at events, mostly PS94); not interictal;")
rep("and does not predict failed or skipped far-contra attempts (23.5); event rates in pre-cue, ITI and quiet windows stay above pre chronically (23.15);",
    "and goes with more failed far-contra attempts only subacutely (23.5); ITI and quiet event rates stay above pre chronically, pre-cue rates ipsilesionally (23.15);")

# ---------------- 23.1 ----------------
set_img("q_fr_c1guard.png", "q_fr_syncend.png")
set_caption("q_fr",
    "<b class=\"h\">Fig 23.1 &middot; NEURAL &middot; face-regressed quiet-period event rate, change vs own pre (headline).</b> Each area trace "
    "(dF/F %, minus its 2-s running median) has face movement regressed out (own-session FaceRhythm factors + 20 point PCs + speeds, "
    "lags &plusmn;0.3 s, out-of-fold ridge fitted on engaged quiet). Event = a peak (up or down) whose height above the median of the "
    "preceding 1.0&ndash;0.2 s exceeds k = 4 &times; the animal&rsquo;s own August-pre robust SD of the residual. Orofacial events / min "
    "(both hemispheres), per animal pre &rarr; acute &rarr; subacute &rarr; chronic: PS92 7.1 &rarr; 36.2 &rarr; 27.8 &rarr; 17.3, "
    "PS93 6.5 &rarr; 50.9 &rarr; 20.6 &rarr; 12.6, PS94 4.7 &rarr; 31.0 &rarr; 17.8 &rarr; 5.5, PS95 3.8 &rarr; 23.1 &rarr; 8.4 &rarr; 5.2. "
    "Change vs own pre per hemisphere (events / min; acute / subacute / chronic): ipsilesional +31.1 [22.2, 42.7] / +11.5 [6.1, 16.7] / "
    "+4.1 [0.6, 9.1] (4/4 each); contralesional +26.4 [16.3, 36.9] / +13.3 [6.9, 21.8] / +4.0 [0.25, 10.0] (4/4, 4/4, 3/4: PS94 "
    "&minus;0.07 chronic); all starred. Mixed model (log10(rate + 0.2), both hemispheres): &times;7.3 acute, &times;3.7 subacute, "
    "&times;1.9 chronic (q &le; .001); raw (not face-regressed) &times;5.5 / &times;3.0 / &times;1.8. Limb (face-regressed, both "
    "hemispheres) per animal pre &rarr; acute &rarr; chronic: PS92 4.2 &rarr; 6.5 &rarr; 24.1, PS93 5.1 &rarr; 34.5 &rarr; 22.0, "
    "PS94 4.6 &rarr; 6.8 &rarr; 7.2, PS95 2.5 &rarr; 11.6 &rarr; 5.1 (large acutely in PS93 / PS95); change vs pre ipsilesional +12.5 "
    "[0.2, 29.1] (3/4) / +2.8 [0.6, 5.3] / +10.2 [2.4, 18.1], contralesional +9.2 [3.5, 17.6] / +5.0 [1.5, 9.8] / +9.2 [1.3, 17.8] "
    "(4/4 each); all starred. The chronic outlier (open circle near +60 to +75) is PS92_0828, which has 2.3 min of task-period engaged "
    "quiet: an unusual task-period session, kept in the statistics. k = 4 separates pre from acute well (Fig 23.18), so the acute "
    "contrast is optimistic; subacute / chronic did not enter the choice. " + STARS + " Face video (FaceRhythm) enters only as "
    "regressors. n = 4 animals, 107 sessions (PS93 08-13 has no face video). " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_spikes_v2_fr.py</code>, restyled by "
    "<code>scripts/cohort2_qc/quiet_restyle_figs.py</code> (<code>restyled_20261118/quiet_events_face_regressed_restyled.png</code>); "
    "same data as PI deck slide 156. PROVISIONAL (n = 4).")

# ---------------- 23.2 ----------------
set_img("q_eta_maps_fr_c1guard.png", "q_eta_maps_fr_syncend.png")
set_caption("q_eta_maps_fr",
    "<b class=\"h\">Fig 23.2 &middot; NEURAL &middot; movement-corrected event-triggered average maps.</b> Face movement regressed out "
    "of every pixel of the high-passed movie (2-s moving-average high-pass) and of the detection trace; UP events &gt; k = 4 &times; the "
    "animal&rsquo;s own August-pre robust SD of the residual. Rows: orofacial UP pre / acute / DELTA (acute &minus; pre within animal), "
    "then limb UP pre / acute / DELTA; mean of animal means (4 animals); per-animal maps not shown. Outline (black) = significant "
    "8&times;8-px bins: nested animals &rarr; sessions bootstrap (2,000 draws), max-statistic family-wise p &lt; 0.05 over 2,022 bins, "
    "eroded mask. Orofacial DELTA: at +0.2 s acute events recruit less frontal and lateral cortex in both hemispheres (610 bins); at the "
    "peak (0 s) an ipsilesional lateral patch is larger (74 bins); at &minus;0.3 s small lateral patches are lower (60 bins, 1.6&times; "
    "rim enrichment &mdash; edge-weighted, treat with care); at +1.0 s frontal cortex, mostly ipsilesional, is higher (331 bins: "
    "pre-stroke events dip slightly below baseline there at +1.0 s, acute events do not); +0.5 s none. Limb DELTA: larger over most of "
    "cortex at the peak (1,259 bins); lateral patches are lower at &minus;0.3 s (256) and +0.2 s (204); 12 bins at +0.5 s and 23 at "
    "+1.0 s (small, treat with care). Selection (Fig 23.7, pre-rebuild run): pre-stroke events without face movement are already local; "
    "the lateral spread came from movement-linked events. Event-triggered decay: Fig 23.21. Image left = ipsilesional. No stars (map "
    "test only). n = 4 animals. " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_eta_fr.py</code> (<code>eta_fr/quiet_eta_maps_fr.png</code>, bin counts "
    "in <code>eta_fr/quiet_eta_maps_fr_significance.csv</code>); same render as PI deck slide 204. PROVISIONAL (n = 4).")

# ---------------- 23.4 ----------------
set_img("q_stillness_c1guard.png", "q_stillness_syncend.png")
set_title("q_stillness",
          "The rise holds at matched face stillness in every bin except the most active, and is largest when the face is still",
          "The acute rise holds at matched face stillness in every bin and is largest when the face is still",
          short="The acute rise holds at matched stillness in every bin")
set_caption("q_stillness",
    "<b class=\"h\">Fig 23.4 &middot; NEURAL + face &middot; event rate by face stillness, change vs own pre.</b> Engaged quiet frames "
    "binned by face speed (FaceRhythm point-PC speed, 2-s smoothed) relative to the animal&rsquo;s own pre median; rate = raw orofacial "
    "events (bilateral, up + down, &gt; k = 4 &times; the animal&rsquo;s own August-pre robust SD) per minute of quiet in each bin (bins "
    "with &lt; 0.5 min in a session left out). Left, rate change vs pre within the same bin (events / min), stillest &rarr; most active: "
    "acute +39.2 [30.4, 49.7], +32.5 [19.7, 49.1], +19.3 [9.2, 34.3], +11.0 [2.8, 24.7] (4/4 each), +5.9 [0.3, 15.0] (3/4; PS92 "
    "&minus;0.04) &mdash; all starred, largest when the face is still; subacute starred in every bin (3/4 stillest, 4/4 others); "
    "chronic starred in the 0.25&ndash;0.5 bin (+10.6 [3.7, 18.7], 4/4) and the 0.5&ndash;1 bin (+6.5 [0.2, 15.2], 3/4), not in the "
    "others. Right, share of engaged-quiet time per bin, change vs pre: acutely the face is stiller (stillest bin +0.14 [0.05, 0.23], "
    "0.25&ndash;0.5 +0.12 [0.07, 0.17]; 1&ndash;2 &minus;0.15 [&minus;0.23, &minus;0.08], most active &minus;0.11 [&minus;0.15, "
    "&minus;0.06]; 4/4 each); chronically less time in the stillest bin (&minus;0.07 [&minus;0.13, &minus;0.02], 4/4). Pre-stroke the "
    "same event type is rare in engaged quiet and much more frequent in the quit period (Fig 23.8, pre-rebuild run): acutely, engaged "
    "quiet looks like pre-stroke disengagement in orofacial cortex. " + STARS + " The time-share panel has no mixed model (stars = "
    "bootstrap only). n = 4 animals. " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_stillness.py</code>, restyled by "
    "<code>scripts/cohort2_qc/quiet_restyle_figs.py</code>; statistics in <code>stillness/lmm_rate_by_bin.csv</code> and "
    "<code>restyled_20261118/star_changes_stillness_decay.csv</code>. PROVISIONAL (n = 4).")

# ---------------- 23.5 ----------------
set_img("q_task_outcome_c1guard.png", "q_task_outcome_syncend.png")
set_title("q_task_outcome",
          "A pre-cue orofacial event goes with a weaker frozen-decoder position code in every epoch, pre-stroke included; it does not predict failed or skipped far-contra attempts",
          "A pre-cue orofacial event goes with a weaker frozen-decoder position code in every epoch, pre-stroke included; it goes with more failed far-contra attempts only subacutely",
          short="A pre-cue event goes with a weaker position code in every epoch")
set_caption("q_task_outcome",
    "<b class=\"h\">Fig 23.5 &middot; NEURAL &middot; does a pre-cue event predict the trial?</b> Event = any orofacial event "
    "(ipsilesional or contralesional, up or down, &gt; k = 4 &times; the animal&rsquo;s own August-pre robust SD) in the trial&rsquo;s "
    "clean pre-cue window (spout in place, before the cue; &ge; 1 s clean, no lick or running within 0.5 s); engaged (canonical) trials. "
    "Per session: mean over event trials of (outcome &minus; the mean of its STRICT stratum: same spout &times; within-session "
    "trial-order quartile &times; clean-time bin; by-spout panels: same spout only); sessions with &lt; 3 event trials left out. The "
    "effect is within session, so each epoch is tested against 0 on its own (no pre baseline is subtracted). (a) Far-contra failed "
    "attempt (incomplete / no attempt / reaction): pre +0.011, acute &minus;0.004 (1/4 each), subacute +0.038 [0.007, 0.067] (4/4, "
    "starred; PS92 +0.023, PS93 +0.034, PS94 +0.026, PS95 +0.051), chronic +0.018 [&minus;0.002, 0.056] (3/4); incomplete vs hit n.s. "
    "in every epoch. All spouts, miss vs hit: pre +0.008 [0.001, 0.020] (3/4, starred), acute +0.004 (n.s.), subacute +0.012 [0.001, "
    "0.021] (3/4, starred), chronic +0.006 (n.s.) &mdash; a small event &rarr; miss link that already exists before the stroke. "
    "(b) Frozen pre-stroke decoder p<sub>true</sub> (DAQ 2-s pre-cue window): pre &minus;0.019 [&minus;0.031, &minus;0.006], acute "
    "&minus;0.020 [&minus;0.038, &minus;0.006], subacute &minus;0.018 [&minus;0.038, +0.001] (dagger only), chronic &minus;0.014 "
    "[&minus;0.032, &minus;0.001]; negative in 4/4 animals in every epoch (acute PS92 &minus;0.006, PS93 &minus;0.010, PS94 "
    "&minus;0.032, PS95 &minus;0.058). The event goes with a weaker pre-cue position code as much before the stroke as after: it marks "
    "a state with a weaker code in every epoch, not a stroke-specific effect. By spout (bottom middle / right): most of the earlier "
    "trial-pooled permutation stars do not hold at the session level (23 of 64 cells change; list in the record). The frame-exclusion "
    "follow-up (event frames removed from the decoder input, Fig 23.23) was run on the pre-rebuild data only. Open markers = sessions "
    "(shape = animal, color = epoch); filled = pooled mean with nested animal &rarr; session bootstrap 95% CI; <b>*</b> = CI excludes 0 "
    "(primary); &dagger; = the earlier trial-pooled within-stratum permutation p &lt; .05 (backup). <b>Sessions:</b> PS93 08-13 has no "
    "behavior video (EXPERIMENT_ERRORS), so it has no video attempt classes or frozen-decoder p<sub>true</sub> and is excluded from "
    "every T13 test: 31 pre sessions (PS93 7), 16 acute, 17 subacute, 43 chronic; the quiet-rate figures keep all 32 pre sessions. "
    "<b>Licks:</b> pre-cue windows exclude frames within 0.5 s of a DAQ contact lick, an LP video lick (cam4 <code>licks_all</code> + "
    "cam1-only events that are non-droplet with med_out &ge; &minus;120 px) or running; attempt classes from trial_attempts.csv (hit = "
    "DAQ spout contact); the decoder reads the DAQ 2-s pre-cue window, and p<sub>true</sub> comes from the production camera-class "
    "attempt_coding tables (no cam1 label enters it). <b>Data:</b> rebuilt base with the cam1 guard, analyzed time ending at the last "
    "sync pulse, " + RUN + ", <code>task_events/t13_trials.csv.gz</code>. Scripts widefield_pipeline "
    "<code>scripts/cohort2_qc/quiet_task_events.py</code> (WIDEFIELD_LICK_SOURCE=daq), restyled by "
    "<code>scripts/cohort2_qc/quiet_restyle_figs.py</code> (session values in <code>restyled_20261118/t13_session_excess.csv</code>); "
    "same data as PI deck slide 158. PROVISIONAL (n = 4).")

# ---------------- 23.6 ----------------
set_img("q_examples_c1guard.png", "q_examples_syncend.png")
set_caption("q_examples",
    "<b class=\"h\">Fig 23.6 &middot; NEURAL &middot; example traces.</b> 60-s windows (middle third of the session, most engaged quiet) "
    "for one pre-stroke and one or two post-stroke sessions per animal; row labels give session and epoch. Trace = area dF/F % minus its "
    "2-s running median, the SAME y-scale in every row (2 % dF/F bar at top left); grey = quiet frames. Small black triangles = events: "
    "peaks (up or down) whose height above the median of the preceding 1.0&ndash;0.2 s exceeds k = 4 &times; the animal&rsquo;s own "
    "August-pre robust SD (1.4826 &times; MAD of the quiet trace), placed just outside the trace and pointing at the peak / trough; each "
    "row&rsquo;s threshold is printed at right. Per animal, orofacial ipsilesional / contralesional thresholds (% dF/F): PS92 1.20 / 1.19, "
    "PS93 1.58 / 1.59, PS94 1.47 / 1.52, PS95 1.88 / 2.05. In PS92&rsquo;s windows: pre (08-11) 1 orofacial event (ipsilesional, up), "
    "none contralesional; acute (08-20) 52 ipsilesional / 57 contralesional (up + down). Bottom: blue ticks = cues, dark bar = spout out, "
    "red = DAQ contact licks, orange = LP video licks, green = running. No statistic (illustration; rates and tests in Fig 23.1). "
    + data_block() + " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_spikes_v2_examples.py</code> "
    "(<code>quiet_event_example_traces.png</code>); the same render as PI deck slide 155. PROVISIONAL (n = 4).")

# ---------------- 23.11 ----------------
set_img("q_hemo415_c1guard.png", "q_hemo415_syncend.png")
set_title("q_hemo415",
          "Haemodynamics excluded: the 415 nm channel shows none of the changes",
          "Hemodynamics: at 415 nm quiet power and event rate do not rise; the 415 response at the events rises slightly (mostly PS94)",
          short="415 nm: no power or rate rise; small response at events")
set_caption("q_hemo415",
    "<b class=\"h\">Fig 23.11 &middot; CONTROL &middot; corrected 470 vs uncorrected 470 vs 415 nm.</b> Same engaged quiet frames and "
    "bilateral orofacial ROI in three channels: hemodynamics-corrected 470 (analyzed), uncorrected 470, and 415 nm (calcium-insensitive). "
    "Columns: quiet robust SD, 1&ndash;4 Hz and 0.5&ndash;1 Hz power, own-channel event rate (k = 4 &times; that channel&rsquo;s own "
    "August-pre robust SD; rate + 0.2), all as fold vs own pre on a log scale (10^(session log10 value &minus; the animal&rsquo;s mean pre "
    "log10 value)); last column = amplitude at the corrected-470 events (change in dF/F %). Corrected 470, change in log10 (acute / "
    "subacute / chronic): SD +0.30 / +0.18 / +0.15 (4/4 each), 1&ndash;4 Hz +0.71 / +0.42 (4/4) / +0.06 (n.s.), 0.5&ndash;1 Hz +0.48 / "
    "+0.28 (4/4) / +0.13 (n.s.), rate +0.76 / +0.48 (4/4) / +0.24 (3/4); per animal 1&ndash;4 Hz acute +0.74 / +0.88 / +0.46 / +0.75 "
    "(PS92 / PS93 / PS94 / PS95). 415 nm: no rise in SD, power or event rate acutely (SD &minus;0.00, 1&ndash;4 Hz &minus;0.17, "
    "0.5&ndash;1 Hz &minus;0.12, rate &minus;0.16; all n.s.), chronically likewise; subacutely its SD is slightly up (+0.04 [0.009, "
    "0.078], 4/4; mixed model n.s.). Uncorrected 470 shows the corrected-470 increases, smaller (acute 1&ndash;4 Hz +0.43, rate +0.52), "
    "and stays up subacutely (power, rate, amplitude). Amplitude at the corrected-470 events: corrected 470 +0.31 [0.18, 0.49] % dF/F "
    "acutely (4/4), back at pre later; 415 nm +0.22 [0.03, 0.48] acute and +0.17 [0.0001, 0.43] subacute (4/4 each, starred; mixed "
    "model n.s.), carried mostly by PS94 (+0.59 / +0.55; PS92 +0.02 / +0.03, PS93 +0.20 / +0.03, PS95 +0.05 / +0.07). So the quiet "
    "power and rate changes are calcium, and a small hemodynamic / motion component at the events cannot be excluded, mostly in one "
    "animal. " + STARS + " The amplitude column&rsquo;s random-quiet-frame null is in <code>hemo415/quiet_hemo_sessions.csv</code> "
    "(not drawn in change units). n = 4 animals. " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_hemo_check.py</code>, restyled by "
    "<code>scripts/cohort2_qc/quiet_restyle_figs.py</code>; corrected-470 and 415 rows are PI deck slide 157 (right). PROVISIONAL (n = 4).")

# ---------------- 23.15 ----------------
set_img("q_task_rates_c1guard.png", "q_task_rates_syncend.png")
set_caption("q_task_rates",
    "<b class=\"h\">Fig 23.15 &middot; NEURAL &middot; events in task windows, change vs own pre.</b> Same detector and threshold (k = 4 "
    "&times; the animal&rsquo;s own August-pre robust SD, either polarity) in three windows: pre-cue = spout in place, before the cue; ITI "
    "= trial stop to the next spout arrival; quiet = the v2 rule (reference). Pre-cue and ITI frames within 0.5 s of a DAQ contact lick, "
    "an LP video lick, running or a swapped frame pair are dropped; engaged part of the session only (before the quit and the last sync "
    "pulse). Change vs pre, events / min per hemisphere, acute (ipsilesional / contralesional, 4/4 each): pre-cue +16.6 / +13.8, ITI "
    "+21.6 / +18.5, quiet +28.9 / +25.1; subacute all starred (4/4). Chronic: ITI +5.8 [0.6, 13.6] (4/4) / +5.1 [0.01, 13.1] (3/4), "
    "quiet +5.8 [0.4, 13.4] / +5.1 [0.05, 12.3] (3/4 each), pre-cue ipsilesional +5.2 [0.1, 13.6] (3/4), all starred; pre-cue "
    "contralesional +5.1 [&minus;0.4, 14.8] (3/4) is dagger only. Pre-cue frames that are themselves quiet frames rise like quiet "
    "(acute +27.0); pre-cue frames that are not quiet rise less (+10.1). Per animal fold vs pre, both hemispheres (PS92 / PS93 / PS94 / "
    "PS95): acute pre-cue &times;2.0 / &times;3.8 / &times;4.6 / &times;6.4, ITI &times;3.7 / &times;7.2 / &times;4.0 / &times;4.8, quiet "
    "&times;5.0 / &times;6.8 / &times;4.1 / &times;5.8; chronic pre-cue &times;3.3 / &times;1.3 / &times;0.7 / &times;1.4 and quiet "
    "&times;3.2 / &times;1.8 / &times;0.9 / &times;1.4 (above pre in 3/4; PS94 not). " + STARS + " n = 4 animals. " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_task_events.py</code> (WIDEFIELD_LICK_SOURCE=daq), restyled by "
    "<code>scripts/cohort2_qc/quiet_restyle_figs.py</code>; per-animal rates in <code>task_events/t12_animal_epoch_rates.csv</code>. "
    "PROVISIONAL (n = 4).")

# ---------------- 23.17 ----------------
set_img("q_fr_detail_c1guard.png", "q_fr_detail_syncend.png")
set_caption("q_fr_detail",
    "<b class=\"h\">Fig 23.17 &middot; NEURAL + face &middot; detail, change vs own pre.</b> Rows: raw event rate; face-regressed event "
    "rate (Fig 23.1); fraction of raw events with face movement within &plusmn;0.3 s (face speed above the session&rsquo;s 90th "
    "percentile of quiet frames); cross-validated face-model R&sup2; on quiet frames. Events: up + down, &gt; k = 4 &times; the "
    "animal&rsquo;s own August-pre robust SD (raw SD for raw counts, residual SD for regressed counts). Orofacial movement-linked fraction "
    "per animal pre &rarr; acute: PS92 0.31 &rarr; 0.08, PS93 0.31 &rarr; 0.20, PS94 0.51 &rarr; 0.15, PS95 0.35 &rarr; 0.13; change "
    "ipsilesional &minus;0.24 [&minus;0.34, &minus;0.14] acute and &minus;0.16 [&minus;0.27, &minus;0.05] subacute, contralesional "
    "&minus;0.15 [&minus;0.28, &minus;0.04] acute (4/4 each), back at pre chronically. Face-model R&sup2;: orofacial ipsilesional "
    "slightly lower acutely (&minus;0.06 [&minus;0.12, &minus;0.01], 4/4; mixed model n.s.); higher chronically in every group "
    "(orofacial +0.19 / +0.15, limb +0.17 / +0.18, 4/4 each). Raw rates: orofacial up in every epoch except contralesional chronic "
    "(+5.0 [&minus;0.01, 12.0], 3/4, dagger only); limb up acutely (ipsilesional +10.3, contralesional +8.6, 4/4), subacutely (3/4, "
    "4/4) and ipsilesionally chronically (+9.5 [1.1, 21.0], 3/4). " + STARS + " n = 4 animals, 107 sessions (PS93 08-13 has no face "
    "video). " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_spikes_v2_fr.py</code>, restyled by "
    "<code>scripts/cohort2_qc/quiet_restyle_figs.py</code>. PROVISIONAL (n = 4).")

# ---------------- 23.18 ----------------
set_img("q_threshold_c1guard.png", "q_threshold_syncend.png")
set_caption("q_threshold",
    "<b class=\"h\">Fig 23.18 &middot; METHOD &middot; threshold.</b> Rows: event rate vs threshold k (&times; the animal&rsquo;s own "
    "August-pre robust SD, 1.4826 &times; MAD; solid = positive, dashed = negative events) per epoch; session-level ROC AUC for pre vs "
    "acute sessions by polarity; sensitivity to injected GCaMP6s-shaped events (1&ndash;6 % dF/F) in pre and acute sessions. Engaged "
    "quiet frames, August sessions. Gray line = the common k = 4 used for orofacial and limb (Priya 11-01, 11-18). The orofacial AUC is "
    "flat across k 3&ndash;5 (positive events 0.97&ndash;0.99, either polarity 0.99&ndash;1.00; 0.988 at k = 4); the leave-one-animal-out "
    "choice for positive events is 3.5 / 4.0 / 3.0 / 4.0 (held out PS92 / PS93 / PS94 / PS95; median 3.75), so k = 4 sits on the flat "
    "part of the curve. No statistic beyond the AUC; n = 4 animals, 108 sessions. " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_spikes_v2.py --summarise --k 4</code> (<code>threshold_choice.png</code>, "
    "<code>threshold_auc.csv</code>, <code>threshold_auc_loao.csv</code>). PROVISIONAL (n = 4).")

# ---------------- 23.21 ----------------
set_img("q_decay_fr_c1guard.png", "q_decay_fr_syncend.png")
set_caption("q_decay_fr",
    "<b class=\"h\">Fig 23.21 &middot; NEURAL &middot; decay of movement-corrected events, change vs own pre.</b> Peak &rarr; half decay "
    "of the event-triggered residual trace (face movement regressed out), per session, UP events &gt; k = 4 &times; the animal&rsquo;s own "
    "August-pre robust SD of the residual; value at +0.5 s / peak as a second measure. Orofacial events: unchanged; per animal pre "
    "&rarr; acute (session medians) PS92 0.110 &rarr; 0.089, PS93 0.091 &rarr; 0.092, PS94 0.082 &rarr; 0.096, PS95 0.114 &rarr; 0.104 "
    "s; change &minus;0.002 [&minus;0.020, 0.020] s acute (2/4), subacute / chronic n.s.; value at +0.5 s unchanged. A few sessions near "
    "0.03 s (mostly PS94) are failures of the half-decay estimate. Limb events decay faster after the stroke: pre &rarr; acute 0.139 "
    "&rarr; 0.095, 0.181 &rarr; 0.101, 0.159 &rarr; 0.115, 0.169 &rarr; 0.117 s; change &minus;0.055 [&minus;0.076, &minus;0.035] / "
    "&minus;0.035 [&minus;0.075, &minus;0.006] / &minus;0.026 [&minus;0.046, &minus;0.005] s (acute / subacute / chronic, 4/4 each); "
    "limb value at +0.5 s higher acutely (+0.08 [0.004, 0.15], 3/4; mixed model n.s.). " + STARS + " n = 4 animals. " + data_block() +
    " Script widefield_pipeline <code>scripts/cohort2_qc/quiet_eta_fr.py</code>, restyled by "
    "<code>scripts/cohort2_qc/quiet_restyle_figs.py</code>. PROVISIONAL (n = 4).")

assert "c1guard_k375_20261117" not in s[s.index('<figure id="f-q_fr"'):s.index('<figure id="f-q_fr"') + 9000]
open(OUT, "w", encoding="utf8").write(s)
print("ok", n0, len(s))
left = [m.start() for m in re.finditer("v2_quiet_lp_c1guard_k375_20261117", s)]
print("k375 refs left:", len(left))

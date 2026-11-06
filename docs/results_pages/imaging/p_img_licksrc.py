"""Lick source in every figure title and caption (Priya 11-01: "please ensure all figure titles / captions are clear about
this" -- LP video licks vs DAQ spout-contact licks).

Two lick sources exist (wfield_local/lp_licks.py):
  DAQ  the spout-contact sensor (lick_analog); the default whenever WIDEFIELD_LICK_SOURCE is unset. Every render under
       grant_figures/epoch/, DAQ_ENL1s/, map_drift_prepre/contact|DAQ_ENL1s/ and the default (untagged) output folders.
  LP   Lightning Pose video licks: cam4 lick onsets + cam1-only events where cam1 exists (sessions without cam1 use cam4
       alone); trial classes from trial_attempts.csv. Renders under grant_figures/LP_licks/ and map_drift_prepre/LP_licks/
       (pre-rebuild folders; the cohort-1 rebuild writes each under its block's run tag, e.g. map_drift_prepre/LP_licks_<tag>/)
       (writeguard refuses an LP run's write anywhere else), plus attempt_coding / attempt_maps (they refuse to run
       without the LP source) and the combined-encoding runs ("licks-combined").
Licks enter even when a window is not lick-aligned: working / engaged trials, the lick-free pre-cue window, hit / miss
and attempt classes, quiet / rest frames. Each figure was traced to its source PNG (provenance/prov_table.tsv,
fig_sources.txt) and the code that made it; the per-figure evidence is in FIGS below and in provenance/lick_source.tsv.

This step, per figure:
  * a tag right after "Fig N.k &middot; " in the caption head and after the number in the title line above the image,
    e.g. [licks: DAQ spout contact];
  * one plain "Licks: ..." sentence in the method part of the caption (before "Reading:", else before "Script:"), only
    where the caption did not already say it.
Run after the last chain step (section 15 was added by hand-edits in v37-v38). Asserts before writing; writes
provenance/lick_source.tsv.
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
PAGE = HERE / "imaging_page" / "imaging_results.html"
TSV = HERE / "provenance" / "lick_source.tsv"

DAQ = "licks: DAQ spout contact"
LP = "licks: LP video (cam4 + cam1)"
LP_REST = "licks: mixed - LP video gate; REST baseline DAQ contact"
LP_DEC = "licks: mixed - LP video trials / window; frozen decoder trained on DAQ-contact trials"
LP_ATT = "licks: mixed - LP video attempt classes / windows; frozen decoder trained on DAQ-contact trials"

S_WORK = "<b>Licks:</b> DAQ spout contact (it defines the working-trial classes)."
S_REST_LP = ("<b>Licks:</b> LP video licks choose the trials{x}; the REST baseline&rsquo;s not-licking frames come from "
             "DAQ spout contact (precomputed quiet-frame masks).")
S_QUIET = "<b>Licks:</b> DAQ spout contact (quiet frames = no DAQ contact lick within 1.5 s, as Fig 15.1)."
S_HAUFE = "<b>Licks:</b> DAQ spout contact (it defines the working trials{x})."
S_CD = "<b>Licks:</b> DAQ spout contact (success trials, the lick-free pre-cue window and the first-lick alignment)."

E_EPOCH = "source grant_figures/epoch/ (default render = DAQ; LP renders go to grant_figures/LP_licks/)"
E_LPMAP = ("source grant_figures/LP_licks/ or map_drift_prepre/LP_licks/; REST baseline = "
           "position_reference_maps.session_rest_svt_timelocal on quiet_periods masks (detect_licks on lick_analog)")
E_CTMAP = "source grant_figures/epoch/, DAQ_ENL1s/ or map_drift_prepre/contact|DAQ_ENL1s/ (default render = DAQ)"
E_LPPC = "source grant_figures/LP_licks/ or map_drift_prepre/LP_licks/; PRECUE reference uses no licks"
E_HAUFE = "source grant_figures/epoch/ render 10-21 (default DAQ)"
E_Q = "quiet frames from behavior_events lick_onsets (DAQ lick_analog), quit point from session_behavior"

# fid: (source, tag, caption sentence or None, evidence)
FIGS = {
    "b_timecourse": ("DAQ", DAQ, None, "epoch_behaviour.py on cohort_session_metrics (DAQ hits); caption states DAQ"),
    "d_refit_precue": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (working-trial classes and the lick-free pre-cue "
                       "window); &lsquo;no contact or LP gate&rsquo; above means no extra lick gate on top of that.",
                       "epoch_5r_refit_by_position_precue_working.png in grant_figures/epoch/, rendered 10-07 (before LP_licks, 10-21)"),
    "d_refit_cue": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (working-trial classes); &lsquo;no contact or LP "
                    "gate&rsquo; above means no extra lick gate on top of that.",
                    "epoch_5r_refit_by_position_cue_working.png in grant_figures/epoch/, rendered 10-07"),
    "d_slowband": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (engaged trials and the lick-free pre-cue window).",
                   "slowband_decode_check.py -> filter_acausality_test.decode -> _trial_features (default DAQ); run 10-19, before LP_licks"),
    "d_within_precue": ("DAQ", DAQ, None, E_EPOCH + "; caption states DAQ"),
    "d_enl1s": ("DAQ", DAQ, None, "source grant_figures/DAQ_ENL1s/ (DAQ, 1 s window); caption states DAQ"),
    "d_acc_cue": ("DAQ", DAQ, S_WORK, E_EPOCH),
    "d_lp_lick": ("mixed", LP_DEC, "<b>Licks:</b> LP video licks (cam4, plus cam1 where it exists) choose the trials and "
                  "start the lick window; the frozen decoder is the stored DAQ-contact model (trained on DAQ-gated "
                  "pre-stroke trials), reused unchanged.",
                  "source grant_figures/LP_licks/epoch/; frozen_models.py: LP_licks reuses DAQ-frozen models"),
    "d_state": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (working trials; the state decoder&rsquo;s licking "
                "segments start at the first DAQ lick).",
                E_EPOCH + "; locomotor_decoder uses behavior_events lick_onsets (DAQ lick_analog)"),
    "d_state_delta": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (state decoder as Fig 3.1).", E_EPOCH),
    "s_per_animal": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (working trials; state decoder as Fig 3.1).",
                     "per_animal_fig.py from grant_figures/epoch sidecars epoch_11amp / 11c / 13 (DAQ)"),
    "d_recovery": ("DAQ", DAQ, S_WORK, E_EPOCH),
    "d_refitgap": ("DAQ", DAQ, S_WORK, E_EPOCH),
    "j_drift": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (engaged licking trials).",
                "june_august_drift.py on grant_kit._collect_7 (default DAQ, writeguard-checked untagged folder); "
                "PS93_0813 present (it has no LP table)"),
    "fig2_attempt_decoder": ("mixed", LP_ATT, "<b>Licks:</b> LP video (cam4, plus cam1 where it exists) defines the "
                             "attempt classes, the 1 s lick-free pre-cue window and the lick alignment; the frozen "
                             "decoder is the stored DAQ-contact model, reused.",
                             "attempt_coding.py refuses to run unless WIDEFIELD_LICK_SOURCE=lp; uses DAQ-frozen decoder"),
    "fig2b_attempt_decoder_epochs": ("mixed", LP_ATT, "<b>Licks:</b> as Fig 5.1: LP video attempt classes and lick "
                                     "alignment; frozen decoder trained on DAQ-contact trials.",
                                     "attempt_coding.py (LP only); DAQ-frozen decoder"),
    "e_amp": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (licking / working trials).", E_EPOCH),
    "e_ceiling": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (licking / working trials).", E_EPOCH),
    "e_captured": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (licking / working trials).", E_EPOCH),
    "map_REST_precue_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "drift_REST_precue_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "map_REST_precue_LP": ("mixed", LP_REST, S_REST_LP.format(x=" and the 1 s lick-free window"), E_LPMAP),
    "drift_REST_precue_LP": ("mixed", LP_REST, S_REST_LP.format(x=" and the 1 s lick-free window"), E_LPMAP),
    "map_REST_enl1s_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "drift_REST_enl1s_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "map_REST_cue_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "map_REST_cue_contact_by_animal": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "drift_REST_cue_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "map_REST_cue_LP": ("mixed", LP_REST, S_REST_LP.format(x=" (working = attempted, by video)"), E_LPMAP),
    "drift_REST_cue_LP": ("mixed", LP_REST, S_REST_LP.format(x=" (working = attempted, by video)"), E_LPMAP),
    "map_PRECUE_cue_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "drift_PRECUE_cue_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "map_PRECUE_cue_LP": ("LP", LP, None, E_LPPC + "; caption Gate line states LP"),
    "drift_PRECUE_cue_LP": ("LP", LP, None, E_LPPC + "; caption Gate line states LP"),
    "map_REST_lick_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "drift_REST_lick_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "map_REST_lick_LP": ("mixed", LP_REST, S_REST_LP.format(x=" and start the lick window"), E_LPMAP),
    "map_REST_lick_LP_by_animal": ("mixed", LP_REST, S_REST_LP.format(x=" and start the lick window"), E_LPMAP),
    "drift_REST_lick_LP": ("mixed", LP_REST, S_REST_LP.format(x=" and start the lick window"), E_LPMAP),
    "map_PRECUE_lick_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "drift_PRECUE_lick_contact": ("DAQ", DAQ, None, E_CTMAP + "; caption Gate line states DAQ"),
    "map_PRECUE_lick_LP": ("LP", LP, None, E_LPPC + "; caption Gate line states LP"),
    "drift_PRECUE_lick_LP": ("LP", LP, None, E_LPPC + "; caption Gate line states LP"),
    "a_maps": ("mixed", LP_REST, "<b>Licks:</b> LP video (cam4, plus cam1 where it exists) defines the attempt classes "
               "and starts the lick window; the rest baseline&rsquo;s not-licking frames come from DAQ spout contact.",
               "attempt_maps.py refuses to run unless WIDEFIELD_LICK_SOURCE=lp; rest baseline = session_rest_svt_timelocal (DAQ quiet masks)"),
    "a_maps_minus": ("mixed", LP_REST, "<b>Licks:</b> as Fig 8.1 (LP video classes and lick window; rest baseline "
                     "frames from DAQ contact).", "attempt_maps.py (LP only)"),
    "c_cim": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (success trials and the first-lick alignment).",
              "cd_trajectories (nolick_decoder.categorize, default DAQ), source grant_figures/epoch/"),
    "c_migration": ("DAQ", DAQ, S_CD, "cd_migration.py on cd_trajectories dumps (default DAQ), source grant_figures/epoch/"),
    "c_window_precue": ("DAQ", DAQ, S_CD, "cd_loo_sessions.py -> grant_figures/epoch/data (default DAQ)"),
    "c_window_cue": ("DAQ", DAQ, S_CD, "cd_loo_sessions.py -> grant_figures/epoch/data (default DAQ)"),
    "c_traj_cue": ("DAQ", DAQ, S_CD, "cd_loo_sessions.py -> grant_figures/epoch/data (default DAQ)"),
    "cp_window_precue": ("DAQ", DAQ, S_CD, "cd_loo_pooled.py on grant_figures/epoch/data/cd_window_projection_loo_sessions_long.csv"),
    "cp_window_cue": ("DAQ", DAQ, S_CD, "cd_loo_pooled.py on grant_figures/epoch/data (DAQ)"),
    "cp_window_lick": ("DAQ", DAQ, S_CD, "cd_loo_pooled.py on grant_figures/epoch/data (DAQ)"),
    "cp_traj_precue": ("DAQ", DAQ, S_CD, "cd_loo_pooled.py on grant_figures/epoch/data (DAQ)"),
    "cp_traj_cue": ("DAQ", DAQ, S_CD, "cd_loo_pooled.py on grant_figures/epoch/data (DAQ)"),
    "cp_traj_lick": ("DAQ", DAQ, S_CD, "cd_loo_pooled.py on grant_figures/epoch/data (DAQ)"),
    "r_crossnobis": ("DAQ", DAQ, S_WORK, E_EPOCH),
    "d_drift": ("DAQ", DAQ, S_WORK, "chronic_drift_vs_reorg.py on grant_kit._collect_7 (default DAQ); CSV has 11 pre "
                "sessions for PS93 (an LP run drops PS93_0813)"),
    "d_enl_states": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (success / miss / stopped classes and the strict "
                     "lick-free window).", "enl_decode.py via _trial_features; PNG dated 09-24, before LP_licks"),
    "d_enl_move": ("DAQ", "licks: DAQ spout contact; video = movement regressors only",
                   "<b>Licks:</b> DAQ spout contact (lick-free window; engaged = a DAQ lick within the response window); "
                   "video enters only as motion-energy regressors.",
                   "enl_movement_multisession.py: dlc_frames.lick_onsets = DAQ lick onsets (daq_trials.decode)"),
    "d_rest15f": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (rest lick buffer, lick-gap matching, working trials).",
                  "rest_frozen_decoder.py, _load_cue_events (DAQ); grant_figures/epoch/, rendered 10-07"),
    "m_dec_time": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (decoder trials, as Fig 4.1).",
                   "imaging_lmm.py on grant_figures/epoch/data/recovery_trajectory_matched_cue_working.csv (DAQ)"),
    "m_dec_beh": ("mixed", "licks: mixed - DAQ decoders; deficit = DAQ contact time on LP-video attempts",
                  "<b>Licks:</b> decoders on DAQ-contact working trials (Fig 4.1); the deficit is the time to the first "
                  "DAQ spout contact on trials the video (LP) scores as attempts (trial_attempts ttc_ms).",
                  "mixed_models.DEFICITS['ttc'] = trial_attempts_sessions.csv ttc_ms_far_contra (DAQ contact on LP attempts)"),
    "m_tmpl_time": ("DAQ", DAQ, "<b>Licks:</b> DAQ spout contact (working trials; lick window from the first DAQ lick).",
                    "imaging_lmm.py on grant_figures/epoch/data/*_sessions_long.csv (DAQ)"),
    "m_trend_time": ("LP", "licks: LP video (encoder runs, cam4 + cam1)",
                     "<b>Licks:</b> ceiling and spread come from encoder runs built on LP video licks (cam4, plus cam1-only "
                     "events; lick window = frames near a cam4 lick peak); DAQ contacts enter those runs only as their own "
                     "kernel.", "position_coding_trend_recovery.py: runs *_licks-combined (combined_encoding LICKS_ENV)"),
    "m_trend_beh": ("mixed", "licks: mixed - LP-video encoder measures; deficit = DAQ contact time on LP attempts",
                    "<b>Licks:</b> ceiling and spread from the LP-video encoder runs (Fig 13.4); the deficit is the time "
                    "to the first DAQ spout contact on video-scored (LP) attempts.",
                    "encoder runs *_licks-combined; deficit ttc from trial_attempts"),
    "m_trend_beh_contact": ("mixed", "licks: mixed - LP-video encoder measures; deficit = DAQ-contact share of LP licks",
                            "<b>Licks:</b> ceiling and spread from the LP-video encoder runs (Fig 13.4); the deficit is "
                            "the share of cam4 LP video licks that register a DAQ spout contact.",
                            "position_coding_trend_recovery.behaviour: licks.csv (cam4 LP) contact column"),
    "haufe_precue": ("DAQ", DAQ, S_HAUFE.format(x=" and the lick-free pre-cue window"), E_HAUFE),
    "haufe_cue": ("DAQ", DAQ, S_HAUFE.format(x=""), E_HAUFE),
    "haufe_lick": ("DAQ", DAQ, None, E_HAUFE + "; caption states DAQ"),
    "haufez_precue": ("DAQ", DAQ, S_HAUFE.format(x=" and the lick-free pre-cue window"), E_HAUFE),
    "haufez_cue": ("DAQ", DAQ, S_HAUFE.format(x=""), E_HAUFE),
    "haufez_lick": ("DAQ", DAQ, None, E_HAUFE + "; caption states DAQ"),
    "haufepa_precue": ("DAQ", DAQ, S_HAUFE.format(x=" and the lick-free pre-cue window"), E_HAUFE),
    "haufepa_cue": ("DAQ", DAQ, S_HAUFE.format(x=""), E_HAUFE),
    "haufepa_lick": ("DAQ", DAQ, None, E_HAUFE + "; caption states DAQ"),
    "q_fr": ("DAQ", "licks: DAQ spout contact; face video = regressors only", None,
             E_Q + "; caption states DAQ"),
    "q_examples": ("DAQ", DAQ, None, E_Q + "; caption states DAQ ticks"),
    "q_fr_detail": ("DAQ", "licks: DAQ spout contact; face video = regressors only", S_QUIET, E_Q),
    "q_threshold": ("DAQ", DAQ, S_QUIET, "quiet_spikes_v2.py busy_frames: " + E_Q),
    "q_spectrum": ("DAQ", DAQ, S_QUIET, "quiet_fluct.py imports quiet_spikes_v2.busy_frames: " + E_Q),
    "q_sdmaps": ("DAQ", DAQ, S_QUIET, "quiet_fluct.py (busy_frames): " + E_Q),
    "q_eta_maps": ("DAQ", DAQ, S_QUIET, "quiet_fluct.py (busy_frames): " + E_Q),
}

h = PAGE.read_text(encoding="utf8")
assert "[licks:" not in h, "already applied"
blocks = re.findall(r'<figure id="f-([^"]+)" data-fid="[^"]+">.*?</figure>', h, re.S)
assert len(blocks) == 83 and set(blocks) == set(FIGS), (len(blocks), set(blocks) ^ set(FIGS))

rows = []
for fid, (src, tag, sent, ev) in FIGS.items():
    assert src in ("DAQ", "LP", "mixed") and tag.startswith("licks: ")
    m = re.search(r'<figure id="f-%s" data-fid="%s">.*?</figure>' % (re.escape(fid), re.escape(fid)), h, re.S)
    assert m, fid
    old = m.group(0)
    new = old
    # caption head
    hm = re.search(r'<b class="h">(Fig (\d+\.\d+)) &middot; ', new)
    assert hm and new.count('<b class="h">') == 1, fid
    num = hm.group(2)
    new = new.replace(hm.group(0), hm.group(0) + f"[{tag}] ", 1)
    # title line above the image
    tm = re.search(r'<span class="fnum">Fig %s</span>' % re.escape(num), new)
    assert tm and new.count('class="ftitle"') == 1, fid
    new = new.replace(tm.group(0), tm.group(0) + f"[{tag}] ", 1)
    # caption sentence, in the method part
    if sent:
        cap = re.search(r"<figcaption>.*?</figcaption>", new, re.S).group(0)
        anchor = " <b>Reading:</b>" if " <b>Reading:</b>" in cap else " <b>Script:</b>"
        assert cap.count(anchor) == 1, (fid, anchor)
        new = new.replace(cap, cap.replace(anchor, " " + sent + anchor, 1), 1)
    assert h.count(old) == 1, fid
    h = h.replace(old, new, 1)
    rows.append((fid, num, src, tag, ev))

assert h.count("[licks: ") == 2 * 83
rows.sort(key=lambda r: tuple(int(x) for x in r[1].split(".")))
PAGE.write_text(h, encoding="utf8")
with TSV.open("w", newline="", encoding="utf8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["fid", "number", "source", "tag", "evidence"])
    w.writerows(rows)
print(f"tagged {len(rows)} figures; sentences added to {sum(1 for v in FIGS.values() if v[2])}; wrote {TSV.name}")

#!/usr/bin/env python
"""Reorder the goal page's figures into Key / Supporting / Supplementary tiers per
section, give every figure a STABLE id, renumber, and rewrite every cross-reference.

Page: _goal_page/index.html (the encoding goal page; untracked, backups on M:).

Stable ids
----------
Every <figure> gets  id="f-<fid>" data-fid="<fid>"  where fid = slug of its first
image file name (or of its title when it has no image).  fids are kept in
_goal_page/fig_ids.csv and are re-read from data-fid on later runs, so they never
change.  Link to a figure with  href="#f-<fid>"  -- that survives any renumbering.

Old numbers keep working: each figure carries empty anchors
<span class="oldnum" id="figS-N"></span> for every number it has had at a reorder.
On later runs the pre-run number of each figure is added; if another figure already
holds that anchor id, the most recent owner (this run's pre-run owner) wins and the
conflict is logged.

Tiers
-----
TIER_OVERRIDES (below, keyed by fid) holds every tier decision.  Figures not in it
fall back to:  caption tag "KEY" -> key;  "supplementary"/"SUPPLEMENT" in title or
caption header -> supplementary;  else supporting.  Within a tier the current order
is kept.

What gets rewritten
-------------------
* fnum span and the "Fig S.N" at the start of the caption header (own number);
* links <a href="#figS-N">..</a> / <a href="#f-fid">..</a>  -> href="#f-<fid>",
  link text = new number (keeps a leading "Fig " / "Figs ");
* plain-text references: numbers after Fig/Figs/fig/Figure(s) and list continuations
  ("Figs 6.1, 6.7, 6.10", "Figs 0.1&ndash;0.13, 1.1&ndash;1.2"), plus bare numbers in
  clear figure contexts ("as 3.15", "see 3.72", "(3.57)", "behind 3.42", "the 3.33
  lateral maps", "&middot; 3.36 with"); section-0 bare numbers are never touched
  (too many 0.x data values) -- they are listed as ambiguous instead;
* ranges whose members are no longer contiguous become explicit lists;
* caption tag tokens that contradict the new tier (" &middot; KEY" on a non-key
  figure, " &middot; SUPPLEMENT(ARY)" on a non-supplementary one) are removed;
* a "the same ..." caption header whose predecessor no longer sits right before it
  gets "(as Fig X.Y)" (a stable link); "(Method: see the lick-window caption below.)"
  gets the figure link when that caption is no longer directly below.
Every replacement outside figure headers goes to the log, plus the bare numbers that
look like possible figure references but were left alone.

Usage
-----
python scripts/goal_page_reorder.py                       # rewrite _goal_page/index.html
python scripts/goal_page_reorder.py --out /tmp/x.html     # dry run to another file
Options: --tag 20261027 --version v129 --date 10-27 (used in outputs / the page note).
"""
import argparse
import csv
import html
import os
import re
import sys
from collections import OrderedDict
from html.parser import HTMLParser

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# TIER DECISIONS (editable).  key / supporting / supplementary, keyed by stable fid.
# Comment = number at the 10-27 reorder (v128 numbering) and short title.
# Section 3 decisions are Priya's (10-27); the rest are first-pass judgement calls.
# ---------------------------------------------------------------------------
K, S, X = 'key', 'supporting', 'supplementary'
TIER_OVERRIDES = {
    # --- 0. frozen pre-stroke encoder: headline ---
    'frozen_fr_compare_lines_cue': K,           # 0.1  FR does not absorb chronic change, cue
    'frozen_fr_compare_lines_lick': S,          # 0.2  same, lick
    'frozen_musall_cue': K,                     # 0.3  unique R2 frozen, cue
    'frozen_musall_lick': S,                    # 0.4  same, lick
    'frozen_claims_contest_cue': K,             # 0.5  target vs executed frozen, cue
    'frozen_claims_contest_lick': S,            # 0.6  same, lick
    'frozen_claims_peth_cue': K,                # 0.7  PETH vs trial-to-trial frozen, cue
    'frozen_claims_peth_lick': S,               # 0.8  same, lick
    'frozen_claims_trend': S,                   # 0.9  spout unique by session vs contact
    'recovery_timing_lick': K,                  # 0.10 recovery timing on hits
    'recovery_timing_pooled_day_lick': S,       # 0.11 recovery pooled by day
    'recovery_timing_pooled_epoch_lick': K,     # 0.12 recovery pooled by epoch: chronic gap
    'early_predictors': X,                      # 0.13 early predictors (exploratory n=4)
    # --- 1. target vs executed (encoder) ---
    'frozen_posinfo_split_cue': S,              # 1.1
    'frozen_posinfo_split_lick': S,             # 1.2
    'frozen_unique_sig_nofr': K,                # 1.3  frozen pixel maps (primary test)
    'frozen_unique_sig_nofr_per_animal': X,     # 1.4  per-animal view
    'spec_groups': K,                           # 1.5  specificity control
    'spec_area_maps': S,                        # 1.6
    'remap_resid_cue_deficit': S,               # 1.7
    'remap_resid_lick_time': S,                 # 1.8
    'remap_layer2_time': S,                     # 1.9
    'matched_residual': K,                      # 1.10 behaviour-matched residual
    'peth_trial_cue': S,                        # 1.11
    'peth_trial_lick': S,                       # 1.12
    'peth_alone_cue': X,                        # 1.13 group-alone layout
    'peth_alone_lick': X,                       # 1.14
    'ls_pixel_alone_cue': X,                    # 1.15 lick-kernel frame alt.
    'ls_pixel_novar_cue': S,                    # 1.16
    'fr_lick_variance': S,                      # 1.17
    'peth_trial_lick_nofr': X,                  # 1.18 robustness without FR
    'kernel_maps': S,                           # 1.19
    'kernel_maps_target6': S,                   # 1.20
    'epoch_kernels_r': X,                       # 1.21 per-epoch kernel refits (untested vs drift)
    'epoch_kernel_maps_lick': X,                # 1.22
    'spout_unique_maps': K,                     # 1.23 per-spout target vs executed maps
    'ls_spout_unique_maps': X,                  # 1.24 lick-kernel frame alt.
    'spout_unique_levels': X,                   # 1.25 levels, descriptive only
    'spout_unique_per_animal_target': X,        # 1.26 per animal
    'spout_unique_per_animal_executed': X,      # 1.27 per animal
    'selffit_cue': K,                           # 1.28 self-fit remapping, cue
    'selffit_lick': S,                          # 1.29
    'remap_maps_cue': S,                        # 1.30
    'remap_maps_lick': S,                       # 1.31
    'remap_per_animal_lick': X,                 # 1.32 per animal
    'remap_time_lick': S,                       # 1.33
    'te_unique_lines_frame': K,                 # 1.34 new design target vs execution kernel
    'te_selffit_lick': S,                       # 1.35
    'te_pixel_parts_farcontra_chronic_lick': X, # 1.36 SUPPLEMENT
    'te_partial_refit_lines': K,                # 1.37 new-design partial refit
    'partial_refit_lines': S,                   # 1.38 old-design partial refit
    'partial_refit_bars_cue': X,                # 1.39 supplement bars
    'contest_landing_vs_target': S,             # 1.40
    'contest_selffit_remap': S,                 # 1.41
    'frozen_unique_sig_fra': S,                 # 1.42
    'frozen_unique_sig_split': S,               # 1.43
    'fig1_binned_contest': X,                   # 1.44 per-session refit contest (superseded)
    'fig1p_contest_pooled': S,                  # 1.45 contest pooled
    'fig10_softmax_width': X,                   # 1.46 width sweep
    'fig10p_width_pooled': X,                   # 1.47 width sweep pooled
    # --- 2. where the licks land (behaviour + decoder vs licks) ---
    'fig9_softmax_landing': S,                  # 2.1
    'fig9b_softmax_landing_change': S,          # 2.2
    'fig9c_softmax_landing_stats': K,           # 2.3  landing change with stats
    'fig9d_path_change_stats': S,               # 2.4
    'fig9e_path_matrices': X,                   # 2.5  per animal
    'fig9g_neural_vs_lick_cue': S,              # 2.6
    'fig9f_neural_vs_lick': K,                  # 2.7  decoder vs landing, lick window
    'fig9h_trial_by_landing': S,                # 2.8
    'fig9i_cls_animals': K,                     # 2.9  coupling by trial class
    'fig9i2_cls_sessions': S,                   # 2.10
    'fig9i3_cls_pooled': K,                     # 2.11 coupling pooled per epoch
    'fig9i4_cls_pooled_norm': S,                # 2.12
    'fig9l_exec_vs_decoder': S,                 # 2.13
    'fig9m_offtarget': K,                       # 2.14 off-target: target or executed?
    'fig9n_offtarget_by_spout': S,              # 2.15
    'exec_dec_far_contra': S,                   # 2.16
    'exec_dec_far_mid': S,                      # 2.17
    'fig9j_relabel_change': S,                  # 2.18
    'fig9k_relabel_matrices': X,                # 2.19 per animal
    'fig3x_relabel_trial_change': X,            # 2.20 encoder counterparts
    'fig3y_exec_vs_encoder': X,                 # 2.21
    'fig3z_encoder_offtarget_by_spout': X,      # 2.22
    'relabel_fc_by_class_cue': X,               # 2.23
    'fig11_lick_landing': S,                    # 2.24
    'fig13_lick_paths': K,                      # 2.25 tongue paths
    'fig13b_farcontra_paths': S,                # 2.26
    # --- 3. far-contra code / decoders (Priya 10-27) ---
    'fig2_attempt_decoder': S,                  # 3.1
    'fig2b_attempt_decoder_epochs': S,          # 3.2
    'fig2c_decoder_pooled': S,                  # 3.3
    'decoder_retention_sessions': S,            # 3.4
    'decoder_retention_pooled': S,              # 3.5
    'precue_class_matched': S,                  # 3.6
    'iti_direction_decoding': S,                # 3.7
    'iti_frozen_timecourse': S,                 # 3.8  (was tagged KEY)
    'iti_frozen_summary': K,                    # 3.9
    'iti_recovery_marker': S,                   # 3.10
    'iti_lick_rate': S,                         # 3.11
    'iti_binned_sweep': X,                      # 3.12
    'iti_binned_summary': X,                    # 3.13 (was tagged KEY)
    'iti_basis_decoder': X,                     # 3.14 (was tagged KEY)
    'iti_where_decoder_z': K,                   # 3.15
    'iti_where_decoder_bases': X,               # 3.16
    'iti_where_decoder_locanmf': X,             # 3.17
    'iti_where_decoder_svdbasis': X,            # 3.18
    'iti_where_scatter_trial_z': S,             # 3.19
    'iti_where_scatter_iti_z': S,               # 3.20 (was tagged SUPPLEMENT)
    'iti_where_delta_z': S,                     # 3.21
    'iti_where_per_spout_z': S,                 # 3.22
    'iti_ramp_peth': S,                         # 3.23
    'iti_ramp_sessions': K,                     # 3.24
    'iti_ramp_summary': S,                      # 3.25 (was tagged KEY)
    'iti_ramp_peth_isolated': S,                # 3.26
    'iti_kinematics_control': S,                # 3.27
    'iti_face_control': S,                      # 3.28
    'iti_area_maps': S,                         # 3.29 (was tagged SUPPLEMENT)
    'iti_area_peth': S,                         # 3.30 (was tagged SUPPLEMENT)
    'iti_area_compare': S,                      # 3.31 (was tagged SUPPLEMENT)
    'iti_area_groups': S,                       # 3.32 (was tagged SUPPLEMENT)
    'iti_decoder_patterns_lat': X,              # 3.33
    'iti_decoder_patterns_animals_lat': X,      # 3.34
    'cued_spout_patterns': X,                   # 3.35
    'iti_where_drift': S,                       # 3.36
    'iti_recovery_link_angle_z': S,             # 3.37 (was tagged KEY)
    'far_contra_cortical_pattern_after_the': X, # 3.38 table (no image)
    'iti_recovery_link_pooled_angle_z': K,      # 3.39
    'iti_recovery_link_sessions_angle_z': S,    # 3.40
    'iti_where_drift_june': S,                  # 3.41
    'iti_check_error_per_lick_iti': K,          # 3.42
    'cued_check_error_per_lick': K,             # 3.43
    'iti_check_error_sessions': S,              # 3.44
    'cued_check_error_sessions': S,             # 3.45
    'iti_check_f0': S,                          # 3.46
    'iti_check_f0_sessions': S,                 # 3.47
    'iti_check_3axis': S,                       # 3.48
    'iti_check_3axis_sessions': S,              # 3.49
    'iti_decoder_pattern_stats_lat': X,         # 3.50
    'iti_decoder_pattern_stats_sag': X,         # 3.51
    'cued_spout_pattern_stats': X,              # 3.52
    'iti_spout_pattern_stats_iti': X,           # 3.53
    'cued_first_lick': S,                       # 3.54 (was tagged KEY)
    'cued_overcorrection': S,                   # 3.55
    'cued_overcorrection_angle': X,             # 3.56
    'iti_recovery_measure': S,                  # 3.57
    'iti_where_angle_reach': X,                 # 3.58
    'iti_check_error_per_lick_iti_angle': X,    # 3.59
    'cued_check_error_per_lick_angle': X,       # 3.60
    'iti_check_error_per_lick_iti_xyz': X,      # 3.61
    'cued_check_error_per_lick_xyz': X,         # 3.62
    'iti_hemo_check_variants': K,               # 3.63
    'iti_hemo_check_edge_vessel': S,            # 3.64
    'iti_hemo_check_state_gap': S,              # 3.65
    'iti_session_pattern_maplevel': K,          # 3.66
    'iti_session_pattern_change_spout_cued': K, # 3.67
    'iti_session_pattern_change_position': S,   # 3.68
    'iti_session_pattern_change_within_spout': S,  # 3.69
    'cued_spout_animal_panels': K,              # 3.70
    'lick_order_iti_cued': K,                   # 3.71
    'lick_order_iti_cued_slope': K,             # 3.72
    'lmm_time_readout': K,                      # 3.73
    'lmm_behaviour_readout': X,                 # 3.74
    'lmm_time_response': K,                     # 3.75
    'lmm_behaviour_response': X,                # 3.76
    'lmm_time_lickclass': S,                    # 3.77
    'lmm_behaviour_lickclass': X,               # 3.78
    'lmm_time_perspout': S,                     # 3.79
    'lmm_behaviour_perspout': X,                # 3.80
    'lmm_time_behaviour': S,                    # 3.81
    # --- 4. failed-attempt maps ---
    'fig3_attempt_amplitude': K,                # 4.1
    'map_attempt_lick': S,                      # 4.2
    'map_attempt_lick_minus_pre': S,            # 4.3
    # --- 5. frozen encoder parts ---
    'fig4_decomp_epochs': S,                    # 5.1
    'fig4p_decomp_pooled': K,                   # 5.2 decomposition pooled
    'map_frozen_acute_lick': S,                 # 5.3
    'map_frozen_subacute_lick': S,              # 5.4
    'map_frozen_chronic_lick': S,               # 5.5
    # --- 6. chronic residual ---
    'fig7_residual_far_contra_16px': S,         # 6.1
    'fig7c_residual_allspouts_32px': K,         # 6.2 PRIMARY pooled test
    'residual_roi_pooled': S,                   # 6.3
    'residual_roi_spout_lick': S,               # 6.4
    'residual_roi_spout_cue': X,                # 6.5
    'fig7d_residual_perspout_cue_32px': X,      # 6.6
    'fig7d_residual_perspout_lick_32px': X,     # 6.7
    'fig7c_residual_allspouts_16px': X,         # 6.8
    'fig7d_residual_perspout_cue_16px': X,      # 6.9
    'fig7d_residual_perspout_lick_16px': X,     # 6.10
    'fig8_residual_per_animal_cue': S,          # 6.11
    'fig8_residual_per_animal_lick': K,         # 6.12 within-animal test
    'fig12_residual_trend_lick': S,             # 6.13
    'fig12p_residual_trend_pooled': K,          # 6.14 trend pooled
    # --- 7. Layer 2 ---
    'fig5_layer2_change': X,                    # 7.1 per animal
    'fig5p_layer2_pooled': K,                   # 7.2
    'fig5b_layer2_random_splits': X,            # 7.3 per animal
    'fig5c_layer2_random_splits_pooled': K,     # 7.4
    'fig6_layer2_groups': X,                    # 7.5 per animal
    'fig6p_layer2_groups_pooled': K,            # 7.6
    # --- 8. variance explained ---
    'fig14_musall_task_aligned': K,             # 8.1
    'fig15b_musall_alone_unique_lines': K,      # 8.2
    'fig15_r2_pre': S,                          # 8.3
    'fig15b_r2_acute': S,                       # 8.4
    'fig15c_r2_chronic': K,                     # 8.5
    'fig15d_r2_pos_pre': S,                     # 8.6
    'fig15e_r2_pos_chronic': S,                 # 8.7
    # --- 9. FaceRhythm ---
    'fig16_fr_cortex': K,                       # 9.1
    'fig16_fr_ssp_m_left': S,                   # 9.2
    'fig16_fr_ssp_m_right': S,                  # 9.3
    'fig16_schematic': X,                       # 9.4 schematic
    'fig16_fr_modes_schematic': X,              # 9.5 schematic
    'fr_musall_cortex_cue_modea': S,            # 9.6
    'fr_musall_cortex_lick_modea': S,           # 9.7
    'fr_pixel_maps_cue_modea': S,               # 9.8
    'fr_pixel_maps_lick_modea': S,              # 9.9
    'fr_pixel_sig_cue': K,                      # 9.10
    'fr_pixel_sig_lick': K,                     # 9.11
    'fr_chronic_per_animal_lick': X,            # 9.12 per animal
    'fr_trial_peth_pre': S,                     # 9.13
    'fr_trial_peth_chronic': K,                 # 9.14
    'fr_direction_areas_cue': S,                # 9.15
    'fr_direction_areas_lick': S,               # 9.16
    'fr_direction_maps_cue': X,                 # 9.17
    'fr_direction_maps_lick': X,                # 9.18
    'fr_direction_mechanism_summary': K,        # 9.19
    'variance_maps_lick': S,                    # 9.20
    'contest_fr_lags': K,                       # 9.21
}

TIERS = [K, S, X]
TIER_HEAD = {K: 'Key figures', S: 'Supporting', X: 'Supplementary'}
TIER_CSS = ('h3.tier{font:600 .78rem/1.2 "Source Sans 3",sans-serif;letter-spacing:.08em;'
            'text-transform:uppercase;color:var(--muted);margin:22px 0 10px;padding-bottom:4px;'
            'border-bottom:1px solid var(--rule)}')

NUM_RE = r'(\d)\.(\d{1,2})(?![\d]|\.\d)'
SEP_RANGE = r'\s*(?:&ndash;|–)\s*'
SEP_LIST = r'(?:,\s*|\s*/\s*|\s+and\s+|\s*&amp;\s*)'
UNIT_AFTER = re.compile(r'\s*(?:%|&deg;|°|&times;|×|s\b|ms\b|mm\b|days?\b|pre-SD|SD\b|bins?\b|'
                        r'areas?\b|px\b|e-fold|Hz\b|x\b|&sigma;)')


def slug(text, maxlen=40):
    s = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    if len(s) > maxlen:
        s = s[:maxlen].rsplit('_', 1)[0]
    return s or 'fig'


def strip_tags(s):
    return html.unescape(re.sub(r'<[^>]+>', '', s)).strip()


class Fig:
    def __init__(self, raw, section):
        self.raw = raw
        self.section = section
        m = re.match(r'<figure\b([^>]*)>', raw)
        self.attrs = m.group(1)
        idm = re.search(r'\bid="([^"]+)"', self.attrs)
        self.elem_id = idm.group(1) if idm else ''
        fm = re.search(r'\bdata-fid="([^"]+)"', self.attrs)
        self.fid = fm.group(1) if fm else None
        nm = re.search(r'<span class="fnum">Fig (\d+)\.(\d+)</span>', raw)
        self.num = (int(nm.group(1)), int(nm.group(2))) if nm else None
        tm = re.search(r'<div class="ftitle">(.*?)</div>', raw, flags=re.S)
        self.title = strip_tags(re.sub(r'<span class="fnum">.*?</span>', '', tm.group(1))) if tm else ''
        hm = re.search(r'<b class="h">(.*?)</b>', raw, flags=re.S)
        self.header = hm.group(1) if hm else ''
        self.imgs = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', raw)
        self.old_anchors = re.findall(r'<span class="oldnum" id="([^"]+)"></span>', raw)
        if self.elem_id.startswith('fig') and re.fullmatch(r'fig\d+-\d+', self.elem_id):
            if self.elem_id not in self.old_anchors:
                self.old_anchors.append(self.elem_id)
        self.tier = None
        self.new = None

    @property
    def numstr(self):
        return '%d.%d' % self.num

    def default_fid(self):
        if self.imgs:
            return slug(self.imgs[0].split('/')[-1].rsplit('.', 1)[0], maxlen=60)
        return slug(self.title)

    def default_tier(self):
        h = self.header
        if re.search(r'&middot; KEY\b', h) or re.search(r'\bKEY\b', strip_tags(h)):
            return K
        if re.search(r'supplement', self.title + ' ' + strip_tags(h), flags=re.I):
            return X
        return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default=os.path.join(REPO, '_goal_page', 'index.html'))
    ap.add_argument('--out', default=None, help='output page (default: overwrite --page)')
    ap.add_argument('--registry', default=os.path.join(REPO, '_goal_page', 'fig_ids.csv'))
    ap.add_argument('--tag', default='20261027')
    ap.add_argument('--date', default='10-27')
    ap.add_argument('--version', default='v129')
    ap.add_argument('--map-out', default=None)
    ap.add_argument('--status-map-out', default=None)
    ap.add_argument('--log', default=None)
    ap.add_argument('--no-side-files', action='store_true', help='only write the page (dry runs)')
    a = ap.parse_args()
    tagdash = '%s-%s-%s' % (a.tag[:4], a.tag[4:6], a.tag[6:])
    out = a.out or a.page
    map_out = a.map_out or os.path.join(REPO, '_goal_page', 'fig_map_%s_reorder.csv' % a.tag)
    status_out = a.status_map_out or os.path.join(REPO, 'docs', 'status', 'GOAL_PAGE_FIG_MAP_%s_REORDER.csv' % tagdash)
    log_path = a.log or os.path.join(REPO, 'docs', 'status', 'GOAL_PAGE_REORDER_LOG_%s.txt' % tagdash)

    src = open(a.page, encoding='utf-8').read()
    log = []
    ambiguous = []

    # ---------------- parse sections and figures ----------------
    figs = []
    regions = []   # (section_id, start, end) of the figure block region in src
    for sec in re.finditer(r'<section id="([^"]+)"[^>]*>(.*?)</section>', src, flags=re.S):
        sid = sec.group(1)
        body_start = sec.start(2)
        sfigs = []
        for fm in re.finditer(r'<figure\b[^>]*>.*?</figure>', sec.group(2), flags=re.S):
            f = Fig(fm.group(0), sid)
            if f.num is None:
                continue          # e.g. the un-numbered "Key figure" in Where things stand
            f.start, f.end = body_start + fm.start(), body_start + fm.end()
            sfigs.append(f)
        if not sfigs:
            continue
        rs, re_ = sfigs[0].start, sfigs[-1].end
        # a tier heading left by an earlier run sits just before the first figure: take it in
        while True:
            hm = re.search(r'<h3 class="tier">[^<]*</h3>\s*$', src[:rs])
            if not hm:
                break
            rs = hm.start()
        # the region may only hold figures, tier headings, whitespace (and stray </figure>)
        rest = src[rs:re_]
        for f in sfigs:
            rest = rest.replace(f.raw, '', 1)
        rest2 = re.sub(r'<h3 class="tier">.*?</h3>', '', rest)
        stray = rest2.count('</figure>')
        if stray:
            log.append('STRUCTURE: removed %d stray </figure> tag(s) between figures in section %s' % (stray, sid))
            rest2 = rest2.replace('</figure>', '')
        if rest2.strip():
            sys.exit('Section %s has non-figure content between its figures: %r' % (sid, rest2.strip()[:300]))
        secnums = {f.num[0] for f in sfigs}
        if len(secnums) != 1:
            sys.exit('Section %s mixes figure numbers from sections %s' % (sid, secnums))
        regions.append((sid, rs, re_, sfigs))
        figs.extend(sfigs)

    by_num = {}
    for f in figs:
        if f.numstr in by_num:
            sys.exit('Duplicate figure number %s' % f.numstr)
        by_num[f.numstr] = f

    # ---------------- stable ids ----------------
    registry = OrderedDict()
    if os.path.exists(a.registry):
        with open(a.registry, encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh):
                registry[row['fid']] = row
    used = {f.fid for f in figs if f.fid}
    for f in figs:
        if f.fid:
            continue
        base = f.default_fid()
        fid, k = base, 2
        while fid in used:
            fid, k = '%s_%d' % (base, k), k + 1
        f.fid = fid
        used.add(fid)
    fidmap = {f.fid: f for f in figs}

    # ---------------- anchors: who owns which id before this run ----------------
    owner = {}
    for f in figs:
        owner[f.elem_id] = f
        owner['f-' + f.fid] = f
        for an in f.old_anchors:
            owner.setdefault(an, f)
    # (owner = how the browser resolves each id BEFORE this run; links are resolved with it)

    # ---------------- tiers, new order, new numbers ----------------
    for f in figs:
        f.tier = TIER_OVERRIDES.get(f.fid) or f.default_tier()
        if f.fid not in TIER_OVERRIDES:
            log.append('TIER DEFAULT (no override): %s %s -> %s' % (f.numstr, f.fid, f.tier))
    old_order = {}
    new_order = {}
    for sid, _rs, _re, sfigs in regions:
        old_order[sid] = list(sfigs)
        ordered = [f for t in TIERS for f in sfigs if f.tier == t]
        new_order[sid] = ordered
        for i, f in enumerate(ordered, 1):
            f.new = (f.num[0], i)
    newstr = lambda f: '%d.%d' % f.new

    # a figure whose number CHANGES keeps its pre-run number as an old-number anchor
    # (most recent owner wins; an unchanged number adds nothing, so re-running is harmless)
    for f in figs:
        if f.new == f.num:
            continue
        an = 'fig%d-%d' % f.num
        for g in figs:
            if g is not f and an in g.old_anchors:
                g.old_anchors.remove(an)
                log.append('ANCHOR CONFLICT: #%s moved from f-%s to f-%s (most recent owner wins)' % (an, g.fid, f.fid))
        if an not in f.old_anchors:
            f.old_anchors.append(an)

    # ---------------- rebuild figures ----------------
    PH = '\x00N:%s\x00'          # placeholder for a figure's new number

    def rebuild(f, sid):
        r = f.raw
        # opening tag: stable id + data-fid + tier; drop stale old-number spans (re-added below)
        r = re.sub(r'^<figure\b[^>]*>', '<figure id="f-%s" data-fid="%s" data-tier="%s">' % (f.fid, f.fid, f.tier), r)
        r = re.sub(r'<span class="oldnum" id="[^"]+"></span>', '', r)
        spans = ''.join('<span class="oldnum" id="%s"></span>' % an for an in f.old_anchors)
        r = r.replace('>', '>' + spans, 1)
        r = re.sub(r'<span class="fnum">Fig \d+\.\d+</span>', '<span class="fnum">Fig %s</span>' % (PH % f.fid), r, count=1)
        hm = re.search(r'<b class="h">(.*?)</b>', r, flags=re.S)
        if hm:
            h = hm.group(1)
            h = re.sub(r'^Fig \d+\.\d+', 'Fig ' + PH % f.fid, h)
            h0 = h
            if f.tier != K:
                h = re.sub(r' &middot; KEY(?= &middot;|\.|$)', '', h)
            if f.tier != X:
                h = re.sub(r' &middot; SUPPLEMENT(?:ARY)?(?= &middot;|\.|$)', '', h)
            if h != h0:
                log.append('TAG REMOVED (contradicts tier %s): Fig %s -> %s [%s]: %r -> %r'
                           % (f.tier, f.numstr, newstr(f), f.fid, strip_tags(h0)[:90], strip_tags(h)[:90]))
            # "the same ..." whose predecessor moved away
            oo, no = old_order[sid], new_order[sid]
            oi, ni = oo.index(f), no.index(f)
            pred_old = oo[oi - 1] if oi > 0 else None
            pred_new = no[ni - 1] if ni > 0 else None
            # only a header whose description STARTS with "the same" (after "Fig N &middot; "
            # and upper-case tag tokens such as "NEURAL &middot; ") refers to the previous figure
            sm = re.match(r'Fig \x00N:[a-z0-9_]+\x00(?: &middot; [A-Z][A-Za-z0-9 /+&;-]*?(?= &middot;))*'
                          r' &middot; (the same\b)(?! \(as )', h, flags=0)
            if sm is None:
                sm2 = re.match(r'Fig \x00N:[a-z0-9_]+\x00(?: &middot; [A-Z][A-Za-z0-9 /+&;-]*?(?= &middot;))*'
                               r' &middot; (The same\b)(?! \(as )', h)
                sm = sm2
            if sm and pred_old is not None and pred_old is not pred_new:
                ins = ' (as <a href="#f-%s">Fig %s</a>)' % (pred_old.fid, PH % pred_old.fid)
                h = h[:sm.end(1)] + ins + h[sm.end(1):]
                log.append('"THE SAME" LINKED: old Fig %s (now %s) header now says "the same (as Fig %s)" = its old predecessor %s, no longer directly before it'
                           % (f.numstr, newstr(f), newstr(pred_old), pred_old.numstr))
            r = r[:hm.start(1)] + h + r[hm.end(1):]
        # "(Method: see the lick-window caption below.)" whose next figure moved away
        oo, no = old_order[sid], new_order[sid]
        oi, ni = oo.index(f), no.index(f)
        succ_old = oo[oi + 1] if oi + 1 < len(oo) else None
        succ_new = no[ni + 1] if ni + 1 < len(no) else None
        if 'the lick-window caption below' in r and succ_old is not None and succ_old is not succ_new:
            r = r.replace('the lick-window caption below',
                          'the lick-window caption, <a href="#f-%s">Fig %s</a>' % (succ_old.fid, PH % succ_old.fid), 1)
            log.append('"CAPTION BELOW" LINKED: Fig %s -> %s points to Fig %s' % (f.numstr, newstr(f), newstr(succ_old)))
        return r

    pieces = []
    pos = 0
    for sid, rs, re_, _sfigs in regions:
        pieces.append(src[pos:rs])
        block = []
        for t in TIERS:
            tf = [f for f in new_order[sid] if f.tier == t]
            if not tf:
                continue
            block.append('<h3 class="tier">%s</h3>' % TIER_HEAD[t])
            block.extend(rebuild(f, sid) for f in tf)
        pieces.append('\n'.join(block))
        pos = re_
    pieces.append(src[pos:])
    doc = ''.join(pieces)

    # ---------------- protect style / script ----------------
    protected = []

    def protect(m):
        protected.append(m.group(0))
        return '\x02P%d\x02' % (len(protected) - 1)
    doc = re.sub(r'<style\b.*?</style>|<script\b.*?</script>', protect, doc, flags=re.S)

    # ---------------- links ----------------
    def resolve_href(h, text=None):
        f = owner.get(h)
        # a legacy-pattern href whose text is exactly its own number ("#fig3-5" / "3.5") written
        # AFTER a reorder means the figure that is 3.5 now, not the one that was 3.5 before
        if text and h == 'fig' + text.replace('.', '-') and text in by_num and by_num[text] is not f:
            log.append('LINK INTENT: #%s with text "%s" resolved to the current %s (f-%s), not the old-number '
                       'anchor owner (f-%s)' % (h, text, text, by_num[text].fid, f.fid if f else None))
            return by_num[text]
        return f

    def fmt_set(members, linkify, prefix_word=None):
        """members: figs in referenced order. Contiguous new numbers -> 'a&ndash;b', else list."""
        news = sorted(members, key=lambda g: g.new)
        idx = [g.new[1] for g in news]
        same_sec = len({g.new[0] for g in news}) == 1
        if same_sec and idx == list(range(idx[0], idx[0] + len(idx))):
            if len(news) == 1:
                return linkify(news[0])
            return linkify(news[0]) + '&ndash;' + linkify(news[-1])
        return ', '.join(linkify(g) for g in news)

    link_ph = []

    def link_token(fig, prefix=''):
        link_ph.append('<a href="#f-%s">%s%s</a>' % (fig.fid, prefix, PH % fig.fid))
        return '\x01L%d\x01' % (len(link_ph) - 1)

    LINK = r'<a href="#((?:fig\d+-\d+)|(?:f-[a-z0-9_]+))">((?:Figs? )?)(\d+\.\d+)</a>'

    def members_between(fa, fb):
        if fa.num[0] != fb.num[0] or fb.num[1] < fa.num[1]:
            return None
        return [by_num['%d.%d' % (fa.num[0], i)] for i in range(fa.num[1], fb.num[1] + 1)]

    def link_range(m):
        fa, fb = resolve_href(m.group(1), m.group(3)), resolve_href(m.group(5), m.group(7))
        if not fa or not fb:
            return m.group(0)
        mem = members_between(fa, fb)
        if mem is None:
            log.append('LINK RANGE NOT A RANGE (links rewritten singly): %s' % strip_tags(m.group(0)))
            return m.group(0)
        first = [True]

        def lk(g):
            # a leading "Fig "/"Figs " stays inside the first link
            pre = m.group(2) if first[0] else ''
            first[0] = False
            return link_token(g, pre)
        res = fmt_set(mem, lk)
        shown = fmt_set(mem, lambda g: newstr(g))
        if html.unescape(strip_tags(m.group(0))) != m.group(2) + html.unescape(shown):
            log.append('LINK RANGE: "%s" -> "%s%s"' % (html.unescape(strip_tags(m.group(0))), m.group(2), html.unescape(shown)))
        return res
    doc = re.sub(LINK + r'(\s*(?:&ndash;|–)\s*)' + LINK, link_range, doc)

    def link_one(m):
        f = resolve_href(m.group(1), m.group(3))
        if not f:
            log.append('LINK UNRESOLVED (left): %s' % m.group(0))
            return m.group(0)
        if m.group(3) != f.numstr and not m.group(1).startswith('f-'):
            log.append('LINK TEXT MISMATCH: href #%s (Fig %s) text %r' % (m.group(1), f.numstr, m.group(3)))
        if m.group(1) != 'f-' + f.fid or m.group(3) != newstr(f):
            log.append('LINK: #%s "%s%s" -> #f-%s "%s%s"' % (m.group(1), m.group(2), m.group(3), f.fid, m.group(2), newstr(f)))
        return link_token(f, m.group(2))
    doc = re.sub(LINK, link_one, doc)

    # other links to figure ids (non-numeric text): retarget only
    def link_other(m):
        f = resolve_href(m.group(1))
        if not f:
            return m.group(0)
        log.append('LINK (non-numeric text, retargeted only): #%s -> #f-%s' % (m.group(1), f.fid))
        return '<a href="#f-%s"' % f.fid
    doc = re.sub(r'<a href="#(fig\d+-\d+)"', link_other, doc)

    # ---------------- plain-text references ----------------
    def ctx(text, s, e, w=55):
        return html.unescape(re.sub(r'<[^>]+>|\x01L\d+\x01', ' @ ', text[max(0, s - w):e + w])).replace('\n', ' ')

    def lookup(sec, n):
        return by_num.get('%s.%s' % (sec, int(n)))

    ITEM = re.compile(NUM_RE + r'(?:' + SEP_RANGE + NUM_RE + r')?')
    SEP = re.compile(SEP_LIST)

    def parse_list(text, p):
        """Parse a figure-number list at p. Returns (end, items) with items = list of member lists."""
        items = []
        end = p
        while True:
            m = ITEM.match(text, p)
            if not m:
                break
            a_ = lookup(m.group(1), m.group(2))
            if a_ is None:
                break
            if UNIT_AFTER.match(text, m.end()):
                break
            if m.group(3):
                b_ = lookup(m.group(3), m.group(4))
                mem = members_between(a_, b_) if b_ else None
                if mem is None:
                    # not a valid range: take only the first number
                    m2 = re.compile(NUM_RE).match(text, p)
                    items.append(([a_], m2.end()))
                    end = m2.end()
                    break
                items.append((mem, m.end()))
            else:
                items.append(([a_], m.end()))
            end = m.end()
            s = SEP.match(text, end)
            if not s:
                break
            p = s.end()
        return end, [it[0] for it in items]

    def render_items(text, start, end, items):
        """Render items (keeping the original separators where possible)."""
        if len(items) == 1:
            return fmt_set(items[0], lambda g: PH % g.fid)
        # several items: if all single numbers, keep separators; else join each item
        seps = re.findall(SEP_LIST, re.sub(NUM_RE + '(?:' + SEP_RANGE + NUM_RE + ')?', '\x03', text[start:end]))
        rendered = [fmt_set(it, lambda g: PH % g.fid) for it in items]
        out_ = rendered[0]
        for sep_, r_ in zip(seps, rendered[1:]):
            out_ += sep_ + r_
        return out_

    STRONG = re.compile(r'(?<![\w-])(?:Figs?|figs?|Figures?|figures?)\.?\s+(?=\d\.\d)')
    WEAK = re.compile(r'(?:(?<=\bas )|(?<=\bAs )|(?<=\bsee )|(?<=\bSee )|(?<=\bbehind )|(?<=\bin )|(?<=\bthe )'
                      r'|(?<=\bof )|(?<=\bmaps )|(?<=versions: )|(?<=basis: )|(?<=&middot; )|(?<=\()'
                      r'|(?<=\x01, )|(?<=\x01 / ))(?=[1-9]\.\d)')
    # "3.39 (per epoch), 3.40 (per session)": a list that continues after a short parenthesis
    CONT = re.compile(r'\s*\([^()<\x01]{1,30}\),\s*(?=[1-9]\.\d)')

    def process_text(text, where):
        out_ = []
        p = 0
        # strong triggers first, then weak; collect spans
        spans = []
        for m in STRONG.finditer(text):
            end, items = parse_list(text, m.end())
            if items:
                spans.append((m.end(), end, items, 'Fig'))
        for m in WEAK.finditer(text):
            if any(s <= m.start() < e for s, e, _, _ in spans):
                continue
            end, items = parse_list(text, m.start())
            if not items:
                continue
            nxt = text[end:end + 1]
            if nxt and not re.match(r'[\s)/,;:.\]]|&', nxt):
                ambiguous.append('%s: %r (followed by %r)' % (where, ctx(text, m.start(), end), nxt))
                continue
            spans.append((m.start(), end, items, 'bare'))
        spans.sort()
        grown = True
        while grown:
            grown = False
            for _s, e_, it_, _k in list(spans):
                c = CONT.match(text, e_)
                if not c or any(s2 <= c.end() < e2 for s2, e2, _, _ in spans):
                    continue
                end, items = parse_list(text, c.end())
                if items and items[0][0].num[0] == it_[0][0].num[0]:
                    spans.append((c.end(), end, items, 'list-continuation'))
                    grown = True
            spans.sort()
        for s, e, items, kind in spans:
            new_txt = render_items(text, s, e, items)
            out_.append(text[p:s])
            out_.append(new_txt)
            p = e
            old_txt = html.unescape(text[s:e])
            shown = re.sub('\x00N:([a-z0-9_]+)\x00', lambda mm: newstr(fidmap[mm.group(1)]), new_txt)
            if html.unescape(shown) != old_txt:
                log.append('TEXT %s [%s]: "%s" -> "%s"   | ...%s...' % (kind, where, old_txt, html.unescape(shown), ctx(text, s, e)))
        out_.append(text[p:])
        res = ''.join(out_)
        # ambiguous: remaining bare numbers matching a figure number in a word context
        for m in re.finditer(r'(?<![\w.+\-−;~=])' + NUM_RE, res):
            sec_, n_ = m.group(1), m.group(2)
            if lookup(sec_, n_) is None:
                continue
            before = res[max(0, m.start() - 12):m.start()]
            if re.search(r'(&minus;|&plusmn;|&asymp;|&ge;|&le;|&lt;|&gt;|[=~\d]|\b[rzpdn]|R&sup2;|\x00)\s*$', before):
                continue
            if UNIT_AFTER.match(res, m.end()):
                continue
            if re.search(r'(FINDINGS|DECISIONS|section|v)\s*$', before):
                continue
            if sec_ == '0' and not re.search(r'\b(as|As|see|See|behind|the|of|in|maps|Fig|Figs)\s$', before):
                continue   # 0.x is almost always a data value
            ambiguous.append('%s: "%s" | ...%s...' % (where, m.group(0), ctx(res, m.start(), m.end(), 50)))
        return res

    parts = re.split(r'(<[^>]+>)', doc)
    cur_where = 'page'
    for i, part in enumerate(parts):
        if part.startswith('<'):
            fm = re.match(r'<figure id="f-([a-z0-9_]+)"', part)
            if fm:
                cur_where = 'caption of pre-run Fig %s' % fidmap[fm.group(1)].numstr
            sm = re.match(r'<section id="([^"]+)"', part)
            if sm:
                cur_where = 'section %s' % sm.group(1)
            am = re.search(r'\balt="([^"]*)"', part)
            if am and re.search(r'\d\.\d', am.group(1)):
                new_alt = process_text(am.group(1), cur_where + ' (alt)')
                parts[i] = part[:am.start(1)] + new_alt + part[am.end(1):]
            continue
        if '\x02P' in part and not part.strip('\x02P0123456789'):
            continue
        parts[i] = process_text(part, cur_where)
    doc = ''.join(parts)

    # ---------------- restore placeholders ----------------
    doc = re.sub(r'\x01L(\d+)\x01', lambda m: link_ph[int(m.group(1))], doc)
    doc = re.sub(r'\x00N:([a-z0-9_]+)\x00', lambda m: newstr(fidmap[m.group(1)]), doc)
    doc = re.sub(r'\x02P(\d+)\x02', lambda m: protected[int(m.group(1))], doc)
    assert '\x00' not in doc and '\x01' not in doc and '\x02' not in doc

    # ---------------- CSS + page note ----------------
    if 'h3.tier{' not in doc:
        doc = doc.replace('</style>', TIER_CSS + '\n</style>', 1)
    note = ('<p class="sub" id="figorder">Figure order (%s, %s): within each section figures run '
            '<b>Key</b> &rarr; <b>Supporting</b> &rarr; <b>Supplementary</b>, so figure numbers changed on %s. '
            'Old numbers still work as links (<code>#figS-N</code>, e.g. <code>#fig3-15</code> jumps to the figure that was 3.15); '
            'stable links that survive any renumbering use <code>#f-&lt;id&gt;</code> (the figure&rsquo;s image name, '
            'e.g. <code>#f-lmm_time_readout</code>). Old&rarr;new map: <code>_goal_page/fig_map_%s_reorder.csv</code>.</p>'
            % (a.date, a.version, a.date, a.tag))
    if 'id="figorder"' in doc:
        doc = re.sub(r'<p class="sub" id="figorder">.*?</p>', lambda m: note, doc, count=1, flags=re.S)
    else:
        m = re.search(r'(<h1>.*?</h1>\s*<p class="sub">.*?</p>)', doc, flags=re.S)
        if not m:
            sys.exit('could not find the page intro paragraph for the order note')
        doc = doc[:m.end()] + '\n' + note + doc[m.end():]

    # ---------------- verify ----------------
    problems, info = verify(src, doc, figs)
    log[:0] = ["SOURCE PAGE CHECK (info): " + i for i in info]

    # ---------------- write ----------------
    with open(out, 'w', encoding='utf-8', newline='') as fh:
        fh.write(doc)
    rows = []
    for sid, _rs, _re, _sfigs in regions:
        for f in new_order[sid]:
            # title as it reads on the NEW page (its own cross-references renumbered)
            tm = re.search(r'<figure id="f-%s"[^>]*>.*?<div class="ftitle">(.*?)</div>' % re.escape(f.fid), doc, flags=re.S)
            title = strip_tags(re.sub(r'<span class="fnum">.*?</span>', '', tm.group(1))) if tm else f.title
            rows.append((f.numstr, newstr(f), f.fid, f.tier, title))
    if not a.no_side_files:
        with open(map_out, 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['old', 'new', 'fid', 'tier', 'title'])
            w.writerows(rows)
        with open(status_out, 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['old (before %s reorder)' % a.date, 'new (%s, %s)' % (a.version, a.date)])
            w.writerows([(r[0], r[1]) for r in rows])
        reg_rows = OrderedDict()
        for fid, row in registry.items():
            reg_rows[fid] = row
        for r in rows:
            reg_rows[r[2]] = {'fid': r[2], 'number': r[1], 'title': r[4], 'tier': r[3]}
        with open(a.registry, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['fid', 'number', 'title', 'tier'], extrasaction='ignore')
            w.writeheader()
            for row in reg_rows.values():
                w.writerow(row)
    with open(log_path if not a.no_side_files else out + '.log.txt', 'w', encoding='utf-8') as fh:
        fh.write('goal_page_reorder %s (%s)\n\n' % (a.version, a.date))
        fh.write('== PROBLEMS ==\n' + ('\n'.join(problems) or 'none') + '\n\n')
        fh.write('== REPLACEMENTS / CHANGES ==\n' + '\n'.join(log) + '\n\n')
        fh.write('== AMBIGUOUS (left unchanged) ==\n' + '\n'.join(ambiguous) + '\n\n')
        fh.write('== ORDER ==\n')
        for r in rows:
            fh.write('%-6s -> %-6s %-14s %-42s %s\n' % r)
    print('figures: %d  regions: %d  log lines: %d  ambiguous: %d  problems: %d'
          % (len(figs), len(regions), len(log), len(ambiguous), len(problems)))
    for p_ in problems:
        print('PROBLEM:', p_)
    print('wrote', out)


def verify(src, doc, figs):
    problems = []
    info = []
    n_old = len(re.findall(r'<figure\b', src))
    n_new = len(re.findall(r'<figure\b', doc))
    if n_old != n_new:
        problems.append('figure count %d -> %d' % (n_old, n_new))
    old_imgs = sorted(re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', src))
    new_imgs = sorted(re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', doc))
    if old_imgs != new_imgs:
        problems.append('img src set changed: missing %s, extra %s'
                        % (sorted(set(old_imgs) - set(new_imgs)), sorted(set(new_imgs) - set(old_imgs))))
    ids = re.findall(r'\bid="([^"]+)"', doc)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        problems.append('duplicate ids: %s' % dup)
    idset = set(ids)
    for h in re.findall(r'href="#([^"]+)"', doc):
        if h not in idset:
            problems.append('broken href #%s' % h)
    nums = re.findall(r'<span class="fnum">Fig (\d+\.\d+)</span>', doc)
    if len(nums) != len(set(nums)):
        problems.append('duplicate figure numbers')
    # numbering per section is 1..n in document order
    sec = {}
    for n in nums:
        s_, i_ = n.split('.')
        sec.setdefault(s_, []).append(int(i_))
    for s_, l_ in sec.items():
        if l_ != list(range(1, len(l_) + 1)):
            problems.append('section %s numbering not 1..n in order: %s' % (s_, l_))
    # caption header number == fnum
    for fm in re.finditer(r'<figure id="f-[^"]+"[^>]*>.*?</figure>', doc, flags=re.S):
        b = fm.group(0)
        n = re.search(r'<span class="fnum">Fig (\d+\.\d+)</span>', b).group(1)
        h = re.search(r'<b class="h">Fig (\d+\.\d+)', b)
        if h and h.group(1) != n:
            problems.append('caption header %s != fnum %s' % (h.group(1), n))

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.errs = []
        VOID = {'img', 'br', 'hr', 'meta', 'link', 'input', 'source', 'wbr', 'col', 'area', 'base'}

        def handle_starttag(self, tag, attrs):
            if tag not in self.VOID:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in self.VOID:
                return
            if tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    t = self.stack.pop()
                    if t not in ('p', 'li', 'td', 'th', 'tr'):
                        self.errs.append('unclosed <%s> before </%s>' % (t, tag))
                self.stack.pop()
            else:
                self.errs.append('stray </%s>' % tag)
    for name, text in (('before', src), ('after', doc)):
        p = P()
        p.feed(text)
        p.close()
        for e in sorted(set(p.errs)):
            (problems if name == 'after' else info).append('html (%s): %s x%d' % (name, e, p.errs.count(e)))
        if name == 'after':
            left = [t for t in p.stack if t not in ('p', 'li', 'html', 'body')]
            if left:
                problems.append('html (after): unclosed at end: %s' % left)
    return problems, info


if __name__ == '__main__':
    main()

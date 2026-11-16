# B12 watch: encoder claims that rest on pre-cue (ENL) position coding (11-04)

**Finding being watched.** Pre-cue position credit falls once spontaneous lick kernels are modelled. On the smoke
sessions, drop:pos_enl in the LP ENL window, without -> with spontaneous kernels:

| session | without | with |
|---|---|---|
| PS93_0814 (pre) | .0072 | .0029 |
| PS93_0818 (acute) | .0044 | .0037 |

Per-animal numbers come from the O2 run of enc_rebuild_20261104b. Main has the spontaneous kernels; main_nolicksall
does not.

**Sources read.** The live goal page (wf_attempt/_goal_page/index.html, 11-02) and the PI deck
(PI_overview_2026-11-02.pptx, 186 slides, with figure_provenance.csv). Read-only inventory, 11-04.

**Code facts.**
- Only the per-session encoder (combined_encoding) scores an "enl" window.
- The frozen encoder scores all / cue / lick only. In cue / lick windows its pos_enl columns are about zero.
- The page's FR encoder runs (enc_v36_full_20261024) predate LICKS_ALL=split and the lick-free enl window.
- The frozen encoder (v7g3_t3d) has NO pre-cue lick kernel, and split_blocks refuses LICKS_ALL. So in its all-frames
  maps, pre-cue lick activity can land in target (pos_enl). **Open question for Priya:** should the frozen encoder also
  get spontaneous lick kernels?

## A. Rest directly on the per-session encoder's ENL readout (will change)

| location | claim | numbers | readout | changes? |
|---|---|---|---|---|
| Page Fig 21.1, ENL row (fig16_fr_cortex) | unique spout-position R2 with / without FR, change from pre | image | fr_encoding_summary: full - drop:position, "enl", cortex | yes |
| Page sec 21 intro | "Chronic pre-cue face movement explains more cortex than pre" | +0.072 [+0.045, +0.102] | fr:full - full, "enl" | likely (FR lick factors now compete with licks_pre) |
| Page Fig 21.1, ENL row, left column | FR added R2 (modes A / B / S / D) | image | fr_gain, "enl" | likely |
| Page Figs 21.5 / 21.6, ENL rows (SSp-m L / R) | same layout | image (the quoted numbers are cue / lick: no change) | fr_encoding_summary | yes for the ENL rows |

## B. Frozen encoder, all-frames target / position columns (change only if the frozen encoder gets licks_pre)

| location | claim | numbers | readout | changes? |
|---|---|---|---|---|
| Page Fig 2.1 / deck slide 90 (frozen_unique_sig_noFR) | target unique barely changes; subacute target / task fall frontally; spout unique rises | 3 bins; 14 / 12 down; 31 up | frozen_pixel_unique_sig, all frames | likely, small |
| Page Figs 2.10 / 2.12 (frA, per animal) | target / spout / task: few chronic bins | - | same | likely, small |
| Page Figs 20.3-20.5 (fig15c / fig15_r2_pre / fig15b) | unique spout / task frontal and lateral | - | frozen_r2_maps, all frames | likely, small (target / spout / task panels only) |
| Page Figs 20.6 / 20.7 (r2_pos_pre / chronic) | unique R2 of the whole spout-specific response | - | frozen_r2_maps position | likely |
| Page Fig 1.1 (spout_unique_maps) | target unique changes little | 16 bins | spout_unique_maps, whole trial | possible (target rows) |

## C. Mention pos_enl but would not change

- Page Fig 3.3: cue window, where pos_enl is about 0.
- Page Figs 1.2-1.4: cue / lick windows.
- Deck slides 89, 92, 94, 108, 136: frozen encoder, cue / lick windows.

## Decoder or model-free pre-cue claims (NOT encoder; listed so they aren't confused with it)

- Page Fig 8.5 / deck 130-131; Fig 8.6; the hypothesis box (Fig 14.21 S5, lick_enl_target_test).
- Page Fig 14.21 S3 / S4 / S10; Fig 23.5 / deck 146.
- Deck 11, 25-26, 162 (decoders).
- Page Fig 14.21 / S1 / S2 and deck 24 / 169 (model-free dF/F).
- Page Fig 14.21 S9 (behaviour).

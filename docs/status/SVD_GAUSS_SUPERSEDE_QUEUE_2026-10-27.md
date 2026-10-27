# Supersede queue: SVD-basis decoder + Gaussian same-place read-out as PRIMARY (Priya, 10-27 late)

Decision: every primary decoder result on the goal page uses the frozen pre-stroke decoder on the animal's pre-stroke
200-component SVD basis (`WFIELD_LICK_DECODER=svd`; per-session refit decoders on the same basis), and every offset uses
the Gaussian same-place like-for-like reference (`scripts/lick_matched_readout.py --ref gauss`; width chosen on
held-out pre sessions, 0.1 mm on the Allen-area decoder, MSE flat 0.05-0.2 mm). Allen-area and binned versions move to
Supplementary. Mixed models: time + behaviour (deficit = weighted signed aim) + epoch, on the same measures.

Page numbers below are v134 (stable ids in brackets). Status: Q = queued, R = running, D = done.

## A. Read-out OFFSETS -> Gaussian same-place on SVD
| fig | stable id | now | to do | status |
|---|---|---|---|---|
| 3.15 | lick_matched_place_readout | binned place, Allen | gauss + SVD (`--ref gauss`, svd) | Q (Allen gauss D) |
| 3.16 | lick_matched_place_curves | binned, Allen | gauss + SVD, bin-significance strip kept | Q |
| 3.60 | lick_matched_place_perspout | binned, Allen | gauss + SVD | Q |
| 3.11 / 3.12 | lick_order_iti_cued(_slope) | beyond-pull, Allen | first/later + order slope on gauss resid, SVD | Q |
| 3.50 | cued_first_lick | beyond-pull, Allen | supplementary (superseded by 3.11 gauss) | Q |
| 3.51 / 3.77 | cued_overcorrection(_angle) | beyond-pull | supplementary (superseded by 3.60) | Q |
| 3.4 / 3.41 / 3.42 | iti_recovery_link_* | beyond-pull vs aim | redo on gauss resid vs aim (SVD); behaviour-model figs already = this regression | Q |
| 3.5 / 3.6 | *_check_error_per_lick | per-lick error + pull triangles | per-lick gauss resid histograms (SVD) | Q |
| 3.29 / 3.30 | iti_where_delta_z, iti_where_per_spout_z | raw error change / beyond-pull, Allen | supplementary; per-spout = 3.60 | Q |
| 3.48 / 3.49 / 3.78 / 3.79-3.82 | 3-axis, polar, angle/xyz checks | beyond-pull, Allen | coordinate family on gauss same-place per coordinate (lateral, sagittal, angle, elevation, reach, x/y/z), SVD | Q |
| 3.61 | lick_decoded_decomposition | Allen | SVD | Q (R in svd chain) |
| 3.62 | cued_misdirected_licks | Allen | SVD + gauss | Q |
| 3.13 / 3.57 / 3.58 / 3.83-3.86 | lmm_* read-out / lickclass / perspout | binned place, Allen | gauss + SVD inputs | Q |

## B. Decoder r (frozen and per-session refit) -> SVD
| fig | stable id | now | to do | status |
|---|---|---|---|---|
| 3.2 | iti_where_decoder_z | Allen z-scored, frozen r per target | SVD frozen r per target (z-scored per component), Allen -> supp | Q |
| 3.1 / 3.24 | iti_frozen_summary, iti_frozen_timecourse | Allen frozen r | SVD | Q |
| 3.23 | iti_direction_decoding | per-session REFIT decoder, Allen | per-session refit on SVD | Q |
| 3.25 | iti_recovery_marker | frozen r vs hit rate, Allen | SVD | Q |
| 3.40 / 3.47 | iti_where_drift(_june) | Allen | SVD (needs June SVD features; June basis projection) | Q |
| 3.63 / 3.64 | iti_binned_* | Allen | SVD | Q |
| 3.27 / 3.28 | where_scatter (cued / ITI) | Allen | SVD; + session-level CI markers on the 1-SD ellipses | Q |
| 3.65-3.68 | basis comparisons | three bases | stay supplementary (already include SVD) | - |

## C. Controls on the new measure
| item | to do | status |
|---|---|---|
| landing + FaceRhythm | Gaussian on landing + linear FR terms at the lick; also + kinematics (reach, speed, elevation) | Q |
| same-target same-place | 'more misses' vs 'each lick read differently' | Q |
| response-size attribution (lick_gain_prediction) | SVD decoder | Q |
| non-linear decoder | after the above | Q (low) |

## Not in scope (different decoder families) -- ask Priya
- Section 2 and 3.19-3.22: trial-level spout classifiers (frozen pre-stroke position decoder on LocaNMF / Allen
  components, retention by trial class). Not lick-position decoders; switching them is a separate pipeline.
- Haufe maps (3.8, 3.53-3.56, 3.69-3.76): already in the SVD basis (per-session self-fit maps are primary).

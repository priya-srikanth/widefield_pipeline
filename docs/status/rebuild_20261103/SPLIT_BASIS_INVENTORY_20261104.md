# Split area basis: inventory of Allen-ROI consumers (11-04, read-only; wf_rebuild at c3cb3c0c)

**Decision (Priya 11-04, via a8).** Every Allen-ROI analysis uses the iti_licks split basis:
- all Allen areas except cortex and whole MOs / MOp;
- ALM and tjM1 in;
- MOs_rest and MOp_rest per hemisphere;
- masks = `iti_licks.roi_masks` (v8: eroded stat mask + lateral rim + ALM front line).

The plain basis stays as a robustness row only.

## Definitions found
- **The split basis itself: `scripts/iti_licks.py`.**
  - `roi_masks` :258-352; `with_rest` :692; `area_index` :711; `AREA_BASIS` :219 (default split).
  - `RAMP_ROIS` :239 still lists whole MOs / MOp.
- **A second "split", `scripts/encoder_split_rows.py`.** Stat mask only, no rim (about 11 % of ALM / tjM1 pixels dropped). It is
  what the frozen-encoder basis twin uses. Under the decision the encoder side moves to v8, so SSp and cortex rows shift too.
- **Decoder ROI features (plain).**
  - Built by `locanmf_position_decoder._build_signal` :54-83: 66 areas on the raw native mask, whole MOs / MOp.
  - Used by the section D frozen decoders (`locanmf_frozen_decoder`) and section G (`poststroke_compare._pooled`).
  - Also used by precue_lickfree, nolick, evoked_amplitude, spatial_reorganisation, hemispheric_dynamics, precue_attribution,
    xsession_generalization, postcue_window_test, precue_significance, encoder_bins_test, roi_activity and nightly_figs.
- **Plain area targets elsewhere.**
  - frozen_encoder (:645-651).
  - allsession_encoder: Layer 2, with an optional `--roi-targets`.
  - specificity_control groups.
  - residual_roi: plain areas are primary, Bonferroni over the Allen areas.
  - combined_encoding: region table, PETH and time-resolved SSE.
  - fr_direction_mechanism: its own SSp-m / MOp means on the full mask.
- **Already split or v8.** iti decoders, lick extracts, s1 table, quiet events.
- **Not Allen.** The state decoder and the epoch decoder families (joint LocaNMF).

## Switch status by consumer
- **Already split (no change):** ITI decoders and lick chain (goal 10-12.x, 14.18 S6-S20), the s1 table, quiet events (imaging
  15.x, goal 23.x).
- **Switch; frozen-model spec change (new ids, refit):** section D Allen-ROI frozen decoders, section G.
- **Switch; panels named by area, so the layout changes:**
  - iti_licks RAMP_ROIS (14.4), iti_hemo_check (14.3 / 14.34 / 14.35), iti_area_maps ORO (14.31; it now double-counts ALM / tjM1
    inside MOs / MOp), lick_correction (14.18 S9), movement_null_plan_maps;
  - frozen_fr_compare / posinfo / te_readout / partial_refit / contest_lines / session_vs_frozen / remap_checks (goal 0.1, 0.7,
    1.2-1.5, 2.3-2.5, 4.x: MOp panels become tjM1 + MOp_rest);
  - fr_direction_mechanism (22.1), fr_encoding_summary (21.x, one figure per area), evoked_amplitude / precue_attribution
    (analysis deck).
- **Encoder region tables:**
  - Region R2 can be done post hoc through the block route (error <= .0008).
  - PETH and time-resolved SSE (fr_trial_peth, 21.4 / 21.11) need a re-run. The O2 rebuild bundle does it: split ROIs go into
    the per-area traces, and the masks are precomputed locally because brain_mask is not available on O2.
- **Frozen encoder.** The alpha search sums over targets, so new targets move cortex unless alphas are held. Use the basis
  twin, or re-run with `--alphas-from`.
- **Imaging Fig 1.3 / 1.4 (slowband_decode_check, PI 11-02 slide 9).** Still on plain `roi_signal`, inconsistent with the s1
  table. Re-run on split sets.

## Needs a decision (to Priya via a8)
1. **Which mask "split" means for the ENCODER rows.** v8 (with the rim, per the decision text) or stat-mask-only (the
   10-28 encoder rule, where parents tile exactly). v8 also moves the SSp and cortex rows.
2. **Groups:**
   - specificity_control OROFACIAL: should it become ALM + tjM1 + MOs_rest + MOp_rest, and how are they weighted?
   - quiet-events orofacial group: should MOs_rest / MOp_rest join it?
   - the hemispheric_intensity MO group;
   - precue_attribution's deliberate MOs-vs-MOp contrast.
3. **residual_roi.** The Bonferroni family changes if split becomes primary (MOs / MOp x2 out, 8 ROIs in).
4. **"cortex" summary rows.** Keep them as readouts, noting that v8 cortex (with rim) is not the same as stat-mask cortex.
5. **Whole MOs / MOp genuinely needed:** roi_masks / with_rest themselves, the ROI-definition QC figures (lick_roi_*,
   rim_check), and the encoder_split_rows additivity checks.
6. **LocaNMF-label analyses cannot simply switch.** Components carry one Allen seed label: reference_family_roi, the locanmf
   OROFACIAL dicts, the enl_movement_removed / null_potent MO groups. They would need re-assignment by footprint.

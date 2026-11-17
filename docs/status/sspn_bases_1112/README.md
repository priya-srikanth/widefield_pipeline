# SSp-n share of the frozen lateral-decoder offset in three bases (11-12; Priya 11-11 follow-up, go-ahead 11-12)

No re-run: `region_shares.py` sums the saved pixel contribution maps c(p) = w(p) dx(p) of `lick_offset_contrib_map`
(glue-out run 11-11: maps_areas / maps_svdF / maps_locanmfFM .npz) over each Allen area's stat-mask pixels (the area
decoder's own masks, `_grid`). share = area sum / total offset (sum over all pixels = beta . dh). Area-basis SSp-n sums
equal terms_areas.csv to 6.5e-8. Run from wf_render_v20 (8aed3a7b), rebuild profile, default lick env.

ITI licks, SSp-n (R + L) share of the offset, pooled (mean contrib / mean total over animals):

| epoch | areas | LocaNMF | SVD |
|---|---|---|---|
| acute | .89 | .61 | .28 |
| subacute | .95 | .56 | .44 |
| chronic | .77 | .65 | .29 |

SSp-n is the single largest area in every basis and epoch except chronic (areas: RSPagl .80 vs SSp-n .77; LocaNMF: RSPd
.75 vs .65). In SVD it shares the offset with neighboring SSp (acute: SSp-bfd .26, SSp-m .17, SSp-ul .12; chronic MOs .23,
SSp-m .20). Per animal (ITI acute) areas 2.17 / .65 / .67 / .64, LocaNMF .80 / .08 / .77 / .49, SVD .69 / .15 / .23 / -.14.
Cued licks: the offset is small (chronic total ~0), so shares are unstable ratios; not interpretable.
Reading: "lateral somatosensory cortex, SSp-n the largest single area" holds across bases; "SSp-n carries the offset" (90 %)
is an area-basis number -- ~60 % in LocaNMF, ~30 % in SVD. Shares of single areas can exceed 1 (opposite-sign areas).
Files: sspn_by_basis.csv, region_shares_long.csv, top_areas_by_basis.csv, top_regions_iti.txt, region_shares_out.txt.

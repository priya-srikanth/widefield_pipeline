# cohort2_qc — first-session QC & hemo-correction analysis (2026-10-29)

One-off analysis scripts behind `docs/COHORT2_QC_2026-10-29.md`. They read the imaging box's E: scratch
(`E:/labcams_data/20261029/...`) and the SVD/hemo products under each session's
`motion_corrected/wfield_local_results/`, so they run on the **imaging box** in the `wfield` env
(`C:/ProgramData/anaconda3/envs/wfield/python.exe`). They are a record of the analysis, not pipeline code.
Figures write to E: and were copied to N: `…/labcams/cohort2/_tmp_qc_figures_20261029/`.

| script | what it answers | key output |
|---|---|---|
| `usability_qc.py` | edge dimming, vignetting, saturation, within-session 470/415 fade | `cohort2_usability_qc.png` |
| `iso_snr.py` | 415/470 absolute counts + shot-noise ΔF/F floor → "raise the 415 LED?" | table (415 brightest; usable as is) |
| `early_fade.py` | fast settling transient vs slow drift in corrected SVTcorr | settle_s=0 (no fast transient) |
| `slow_spectrum.py` | cumulative variance vs cutoff period → "is 43 min too slow?" | yes; drift at 3–15 min |
| `var_bands.py` | absolute variance by period band (where each session's variance lives) | PS118 fast-dominated; PS120 tiny |
| `perpixel_snr.py` | per-pixel ΔF/F amplitude vs cortex-mean (fair SNR for GCaMP8s) | PS120 at shot floor; pp≈cm |
| `slow_is_bleach.py` | does the surviving slow band track the raw bleach? | no (|r|≤0.22) → not clean residual |
| `stage_traces.py` | raw 470/415 → corrected, per session (3 rows) | `cohort2_stage_traces.png` |
| `processing_stages.py` | full `meegkit_hpfit` cascade (raw→SVD+dF/F→HP→detrend→corrected) | `cohort2_processing_stages.png` |
| `fit_vs_output_demo.py` | acausal high-pass (fit) vs polynomial detrend (output) | `fit_vs_output_demo.png` |

Session paths and the 2_H_W dims are hardcoded for this session; adapt for later cohort2 dates.

# Cross-area population dynamics: literature scan and options for this dataset (2026-10-30)

Priya 10-30: "look at the literature of reporting cross-area neural population dynamics ... Karunesh Ganguly's papers
... cross-area synchronization occurs (and is important for) recovery". Web-verified by a helper unless marked
[unverified]. Context: CURBD (FINDINGS 4.14), region MVAR (4.15) and pooled MVAR (4.16) all failed their stability gates;
what we have that stands is noise correlation / synchrony (14.18 S17), interhemispheric tilt (S10) and S->M xcorr /
Granger (S18).

## Ganguly lab (spikes / LFP)
| Paper | Measure | Result | Cross-area? |
|---|---|---|---|
| Ramanathan 2018 Nat Med 24:1257 | reach-locked low-frequency oscillations (LFO, 1.5-4 Hz), spike-LFO coupling, perilesional M1 | LFOs lost after stroke; their return tracks recovery; LFO-timed stimulation improves reach | no |
| Khanna 2021 Cell 184:912 | ensemble co-firing = shared-over-total variance of a 1-factor FA ("1D-SOT"); reach-locked ITPC 1.5-4 Hz | co-firing and LFO phase consistency rise with recovery and with low-freq stimulation; PAIRWISE correlations did NOT change | no (one area) |
| **Guo, Kondapavulur, Lemke, Won, Ganguly 2021 Cell Rep 36:109370** (the M1-DLS stroke 'alignment' paper Priya means) | CCA between perilesional motor cortex and DLS spiking, 100 ms bins, -1.5..+1.5 s around pellet touch, trials concatenated, TOP canonical correlation per session (units subsampled to a fixed count per animal); plus local single-trial-to-template reliability | max canonical correlation rises over rehabilitation (LME, n = 8 rats) and predicts success rate; top component becomes movement-specific on single trials; LFP coherence was tried first and FELL, so they used CCA [CCA details from a fetched summary, Methods not read verbatim] | yes (cortex-striatum, recovery) |
| Ganguly, Khanna, Morecraft, Lin 2022 Neuron 110:2363 | review | co-firing as the substrate / target after stroke | - |
| Lemke 2019 Nat Neurosci 22:1122 | M1-dorsolateral striatum 3-6 Hz LFP coherence around movement | coherence rises with skill learning (0.18 -> 0.24); tracks movement consistency | yes (M1-DLS) |
| Veuthey 2020 Nat Commun 11:4057 | CCA between M2 and M1 populations -> single-trial cross-area latents | cross-area reach modulation grows with learning, predicts RT / duration; M2 leads M1 | yes |
| Lemke 2021 eLife 10:e64303 | M1-DLS spindle / slow-oscillation coupling in NREM | coupling grows over learning | yes (sleep) |
| Kim 2023 Nature 613:103 | PFC-M1 slow-oscillation coupling in sleep; shared-over-total variance | PFC-M1 coupling up as skill stabilises | yes (sleep) |
| Lemke 2024 Curr Biol 34:1831 | feature-specific information transfer (M1 <-> striatum) | direction of MOVEMENT-SPECIFIC information flow reverses with learning; raw co-activation / total transfer miss it | yes, directed |

**THE 2026 sensory-motor preprint (Priya sent the PDF 10-30): Heimbuch, Khanna, Novik, Morecraft, Ganguly, "Shared
somatosensory-motor neural population dynamics track motor recovery after stroke", bioRxiv 10.64898/2026.02.10.705199.**
3 macaques, M1 hand/arm lesion; 64-ch arrays in PMd and Area 2; reach-to-grasp. Method: **DLAG** (Gokcen 2022, two-area
GPFA) on single trials, 2-s epochs centred on pellet touch, 20 ms bins, sqrt, demeaned by trial average per unit, z-scored;
5 cross-area factors (CFs) + 5 within-area factors (WFs) per area, fixed for cross-session comparison; low spiking-band
power used across recovery (unit identity changes). Readouts per session: CF Gaussian-process **timescale tau**; CF
**delay** (direction); **stereotypy** = per-trial r of the single-trial CF trace vs the leave-one-out mean of the other
trials (kinematically matched trials, +-0.1 s, time-warped), session median (Guo 2021 metric); **velocity consistency** =
|dCF/dt - mean| at reach / touch / retract. Results: recovered animals: CFs explain little variance (PMd 2-6 %, Area 2
7-26 %) but decode wrist speed better than WFs (R2 0.19-0.77), CF tau longer than WF tau (162 vs 117 / 39 ms), PMd leads
Area 2. Over recovery CF tau lengthens (LME t = 2.3) and CF stereotypy / consistency rise with performance (rho -0.38 /
0.29, 3 animals; LME on the 2 animals with a recovery curve); **WFs do not track recovery**. Caveats they state: n = 3
(2 with a recovery curve), correct trials only, sensory feedback vs coordination not separated.

## Methods (general)
Communication subspace / reduced-rank regression (Semedo 2019 Neuron); CCA (Veuthey 2020); DLAG / mDLAG (delayed shared
latents, Gokcen 2022 [journal unverified], NeurIPS 2023); CURBD (Perich, bioRxiv 2020; no journal version found);
MR-SDS (Karniol-Tambour, ICLR 2024; applied to an 8-region widefield dataset). Widefield: Makino 2017 Neuron (M2-led
cortex-wide flow emerges with learning), MacDowell & Buschman 2020 Curr Biol (motifs), Pinto 2019 Neuron (task demand
decorrelates cortex), Vanni 2017 J Neurosci (GCaMP6s at 30 Hz resolves 0.1-1 Hz and 2-5 Hz connectivity modules).

## Stroke connectivity in rodents
Mostly DECREASED homotopic / interhemispheric connectivity that recovers: Bauer 2014 NeuroImage (tMCAO; also warns that
global-signal regression distorts FC), Cramer 2019 NeuroImage (GCaMP widefield, M1 photothrombosis), van Meer 2010
J Neurosci (fMRI), Hakon 2018, Bice 2022 eLife, Allegra Mascaro 2019 Cell Rep; Lim 2014 J Neurosci: network-wide
depression at 1 week; Cecchini 2021 PLoS Comput Biol: slower, irregular propagation subacute. The only "more coupled"
precedent found: van Meer 2012 J Neurosci (subacute excess clustering). **No widefield mouse paper found reporting a
cortex-wide correlation rise at rest after stroke**, so our S17 rest result is either new or a confound (arousal /
global mode / haemodynamics) and needs those controls before it is framed.

## Options for this dataset (ranked)
0. **Heimbuch-style analysis on our data (added 10-30 after reading the preprint).** Two "areas" = sensory and motor groups
   per hemisphere; population dims = the 4 spatial sub-units per region already built by `curbd_pilot.prep` ("multi":
   16 per group) or pixel PCs per group. Epochs +-1 s around the first contact lick (their pellet touch), 33 ms bins (or
   the 3-frame 10 Hz bins). Readouts per session: CF vs WF stereotypy (leave-one-out template r; model-light, can also be
   run on CCA components first), CF delay (S vs M lead; compare S18), CF tau (CAVEAT: GCaMP6s decay sets a floor on
   tau, so deconvolve and read tau changes only within animal), CF vs WF decoding of tongue kinematics (3-D tongue
   speed / angle, our analogue of wrist speed). Test: does CF (not WF) stereotypy track hit rate / far-contra aim across
   recovery, per animal. DLAG reference code is MATLAB (Gokcen); a Python port would need checking / writing (no pip
   installs into the shared env). Stability gate written first, as for MVAR.
1. **Lick-locked CCA / reduced-rank regression between sensory and motor groups** (Veuthey / Semedo): per session,
   cross-validated canonical correlation (or RRR predictive R2 and dimension) of sensory -> motor, on pixel PCs within
   each group or on the area traces; single-trial cross-area latents vs lick latency / aim. Static (no lag model), so it
   avoids what broke MVAR / CURBD; a lagged version (sensory at t - tau -> motor at t) gives direction.
2. **Shared-over-total variance (Khanna 1D-SOT) of area traces**, rest vs licks, per network: our synchrony number in
   the language of the papers Priya cites; separates a shared mode from pairwise structure.
3. **Lick-locked phase consistency (ITPC) and inter-area coherence, 0.1-1 and 1-4 Hz**: the widefield analogue of
   Ramanathan / Khanna / Lemke 2019. GCaMP6s attenuates above ~3 Hz, so phase measures not power, and check against
   deconvolved traces.
4. **Direction as information, not influence** (Lemke 2024): does lick-DIRECTION information appear in motor areas
   after it appears in sensory areas (time-resolved decoding lag, or feature-specific transfer)? Builds on our decoders.
5. MR-SDS or propagation / motif analysis: possible but high-risk at n = 4.

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
| Ganguly, Khanna, Morecraft, Lin 2022 Neuron 110:2363 | review | co-firing as the substrate / target after stroke | - |
| Lemke 2019 Nat Neurosci 22:1122 | M1-dorsolateral striatum 3-6 Hz LFP coherence around movement | coherence rises with skill learning (0.18 -> 0.24); tracks movement consistency | yes (M1-DLS) |
| Veuthey 2020 Nat Commun 11:4057 | CCA between M2 and M1 populations -> single-trial cross-area latents | cross-area reach modulation grows with learning, predicts RT / duration; M2 leads M1 | yes |
| Lemke 2021 eLife 10:e64303 | M1-DLS spindle / slow-oscillation coupling in NREM | coupling grows over learning | yes (sleep) |
| Kim 2023 Nature 613:103 | PFC-M1 slow-oscillation coupling in sleep; shared-over-total variance | PFC-M1 coupling up as skill stabilises | yes (sleep) |
| Lemke 2024 Curr Biol 34:1831 | feature-specific information transfer (M1 <-> striatum) | direction of MOVEMENT-SPECIFIC information flow reverses with learning; raw co-activation / total transfer miss it | yes, directed |

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

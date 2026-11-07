# State-space view of stroke recovery: literature (2026-11-07)

Question (Priya 11-07): BCI work asks how population activity changes within or outside an intrinsic manifold when the
task mapping changes (rotation, scaling, reassociation, re-aiming, local search, new patterns). Ask the same of cortical
widefield activity across pre / acute / subacute / chronic after the VLS (striatal) stroke: is recovered activity the
pre-stroke pattern, a scaled or rotated version, or new, and how does it get there?

Citation status: **verified** = checked against PubMed / PMC / journal on 11-07; **memory** = known to exist, details
not re-checked; **UNVERIFIED** = could not confirm. Check "memory" items before quoting.

## 1. Kinds of change and the test that defines each

| change | defining test | key papers |
|---|---|---|
| within-manifold (fixed repertoire) | post activity inside the pre top-k subspace; learning easy within, hard outside | Sadtler 2014 Nature (memory) |
| reassociation | same repertoire (post points inside pre density) but condition labels remapped; cross-condition match matrix off-diagonal | Golub 2018 Nat Neurosci (verified, PMC5876156); Hennig 2018 eLife (verified) |
| re-aiming (global rotation) | post pattern for condition i ~ pre pattern for neighbor j along an ordered axis; global > local tuning change | Jarosiewicz 2008 PNAS (verified, PMC2614787); Chase, Kass & Schwartz 2012 J Neurophysiol (verified, PMC3404791); Zhou et al. 2019 J Neurophysiol (verified, PMC6485743: fast global, slow local) |
| uniform shift / translation | condition-independent mean displacement; can persist after behavior recovers ("memory trace") | Sun et al. 2022 Nature (verified, PMC9851374; code xlsun79/TDRandUniformShiftAnalyses); Losey et al. 2024 Curr Biol (verified, PMC11097210: residual change in output-null dims) |
| output-null vs potent change | regress output area on input; track change in null vs potent dims | Kaufman 2014 Nat Neurosci (memory); Perich, Gallego & Miller 2018 Neuron (verified, PMC6250582) |
| gain / scaling | same subspace, per-dimension variance changes; diagonal scaling fits as well as full Procrustes | Lim et al. 2014 J Neurosci (verified; mesoscale stroke: network-wide depression with roughly uniform scaling, peri-infarct exceptions) |
| new patterns (outside manifold) | post variance outside pre top-k exceeds the pre-vs-pre split floor; low pre density at post points | Oby et al. 2019 PNAS (verified); Cecchini et al. 2021 PLoS Comput Biol (verified, PMC8159272: new propagation patterns only with combined rehab) |
| exploration then consolidation | rise in trial-to-trial / private variance or dimensionality, then contraction | Athalye et al. 2017 Neuron (verified); Kim et al. 2023 Nature (verified, PMC9812770) |
| preserved latent dynamics, drifting readout | per-session PCA, align across days by CCA; aligned dynamics stable, decoder transfers | Gallego 2018 Nat Commun (verified, PMC6185944); Gallego 2020 Nat Neurosci (verified); Degenhart 2020 Nat Biomed Eng (verified); Safaie 2023 Nature (verified, PMC10665198); review Dabagia, Kording & Dyer 2023 Nat Biomed Eng (verified) |

Geometry tools: alignment index (Elsayed et al. 2016 Nat Commun, verified, PMC5095296: fraction of dataset 2's top-k
variance captured by dataset 1's top-k PCs, vs a random-subspace null drawn from the data covariance); tensor-shuffle /
maximum-entropy surrogates (Elsayed & Cunningham 2017 Nat Neurosci, verified) to test whether a structure is a byproduct
of simpler per-region means and covariances.

## 2. Stroke / injury work with a population framing

- **Ganguly lab.**
  - Ramanathan et al. 2018 Nat Med (verified): reach-related low-frequency activity is lost after stroke and returns
    with recovery.
  - Khanna et al. 2021 Cell (verified, PMC7935019): low-frequency stimulation raises ensemble co-firing and dexterity.
  - **Guo et al. 2021 Cell Rep (verified, PMC8357409):** recovery brings reliable single-trial ensemble activation in
    perilesional cortex AND dorsolateral striatum, with more cross-area alignment. This is the closest to a
    corticostriatal question.
  - Choi et al. 2025 Neuron (verified via commentary, PMC12916029): M1 lesion blurs pre-movement vs movement states in
    PMd factor space; recovery restores the separation.
  - **Heimbuch et al. 2026 bioRxiv (preprint, verified, PMC12919233):** cross-area (PMd-Area 2) factors become longer
    and more stereotyped with recovery; within-area factors do not change consistently. n = 3, mixed models with an
    animal random effect: a template for small n.
- **Mouse mesoscale.**
  - Lim et al. 2014 (above).
  - Cecchini et al. 2021 (above).
  - Bice et al. 2022 eLife (verified).
  - Bandet et al. 2024 eLife (verified): assembly architecture largely preserved away from the infarct.
  - Latifi et al. 2020 Cereb Cortex (verified).
- **Intact widefield latent structure.**
  - MacDowell & Buschman 2020 Curr Biol (verified, PMC7375907): ~14 motifs explain ~75 % of variance and generalize
    across animals.
  - Musall 2019 Nat Neurosci (memory).
  - LocaNMF: Saxena 2020 PLoS Comput Biol (memory).
- **Gaps.** We found no published manifold or latent-dynamics analysis of cortex after a STRIATAL lesion, and no
  widefield cortical dynamics after a striatal lesion. Our design appears to be new.

## 3. What transfers to our data, and pitfalls

Our data are cortical widefield imaging, the lesion is subcortical (the cortex under the window is intact), there are
4 animals, and every session is registered to the Allen atlas.

- **Shared coordinates come for free.** Atlas regions, and pixels after registration, are the same dimensions in every
  session (and in every animal for regions). So subspaces can be compared directly; CCA / Procrustes is a second step
  that asks how much rotation is needed beyond identity. LocaNMF fit per session is NOT matched across sessions: fix
  the components from pre sessions and project every session onto them, or use atlas regions.
- **Gain can look like rotation.** Fit a model ladder:

  identity → global gain → per-region diagonal gain → gain + offset → full linear / Procrustes

  and report how much held-out variance each level adds. With a subcortical lesion, a regional gain change is a
  biological result (loss of basal ganglia → thalamus → cortex drive, damage to corticostriatal endings), not an
  imaging loss. It is one of the hypotheses, not a nuisance to remove.
- **Imaging gain drift.** Window clarity, GCaMP expression and focus change over weeks, and the hemodynamic
  correction may differ after stroke. The pre-vs-pre noise floor covers only the pre span. Use spontaneous / ITI
  variance as a gain reference.
- **Normalization.**
  - z-scoring per session removes gain, so use it for geometry questions.
  - Keep pre-fixed constants for gain questions.
  - Soft-normalize so low-signal regions don't inflate dimensions.
- **Noise floor and trial counts.**
  - Every metric needs a pre-vs-pre split (odd / even trials, session vs session).
  - Subsample pre to the acute trial count per spout.
  - Use cross-validated PCA. Principal angles are biased toward "different" with noisy trial averages.
- **Behavior confound.** The acute aim bias and missing far-contra licks change the conditions themselves. Use working
  trials with the lick gate, and compare kinematically matched trials; otherwise "reassociation" is just a different
  movement.
- **Statistics with n = 4.** Mixed models with an animal random effect, the per-animal sign (k / 4), sessions shown as
  points.

## 4. Candidate analyses (unit: X = (6 spouts x T bins) x regions, trial-averaged, per session)

1. **Manifold preserved?** Variance of each session's X captured by the pre top-k PCs (k by cross-validation on pre),
   vs the pre-vs-pre floor.
2. **Gain vs geometry.** X_post ~ D X_pre + b with D diagonal, against full Procrustes. Is the acute change
   concentrated in the ipsilesional tjM1 / ALM / motor gains? Does D return to 1 with behavior?
3. **Uniform shift (Sun 2022).** Condition-independent displacement (cue / pre-lick). Does it persist chronically after
   behavior recovers (a "memory trace", Losey 2024)?
4. **Reassociation / re-aiming.** 6 x 6 cross-validated similarity, post spout i vs pre spout j. The ipsilesional aim
   bias predicts far-contra post ~ pre far-mid / far-ipsi (off-diagonal toward ipsi), returning to the diagonal with
   recovery. Ties directly to kinematics Figs 4.12-4.19 (licks landing like far mid).
5. **Spout-coding subspace vs condition-invariant.** dPCA / TDR; principal angles of the coding subspace across epochs
   vs subsampled pre-vs-pre.
6. **Exploration then consolidation.** Trial-to-trial variance, participation ratio and single-trial reliability per
   spout over days, at matched trial counts.
7. **Cross-hemisphere / cross-area coupling.** CCA or reduced-rank regression, ipsi- vs contralesional or frontal vs
   sensory (Guo 2021, Heimbuch 2026).
8. **Decoder transfer with corrections.** Train on pre and test on each session raw, after diagonal-gain correction,
   and after Procrustes; the accuracy each correction recovers names the change type (extends the existing frozen
   decoder).

# Literature review: putting post-stroke sessions into a frozen decoder's frame (2026-11-04)

Scope: how others normalize or align sessions before applying a fixed decoder across days, weeks or injury, and
how this maps onto our frames. Frame tags follow `docs/status/CENTRE_MIXSHIFT_2026-11-04.md` and
`docs/status/REST_CENTRE_2026-11-04.md`:

- **W**: label-free bal_em center;
- **Wt**: true-label balanced center;
- **N**: pooled pre-stroke frame;
- **Z**: per-session z;
- **R5 / RW**: rest centers;
- reference-condition anchor;
- intercept-only refit.

**Sourcing.** Every paper below was opened. Each reference is marked:

- **[full text]**: methods were read, from PMC/Europe PMC full text, the publisher PDF or the arXiv HTML;
- **[abstract only]**: only the abstract or metadata could be read.

Statements tagged *our reading* are my inference, not the authors' claim. I found no source for several things.
Those are stated as gaps, not filled in.

---

## 1. The question

A spout-position decoder is trained on pre-stroke widefield sessions and applied frozen to post-stroke sessions. Each
new session has to be put into the decoder's input frame, because F0, SNR, drift and hemodynamics differ between
sessions.

Any per-session center or scale built from that session's own data has two problems:

- **Circularity.** It can absorb real post-stroke change.
- **Trial-mix dependence.** It depends on the session's trial mix: label shift, missing classes, and the share of
  failed or incomplete attempts.

So the question is which frame to use, how others solved the same problem, and how to report how much the
conclusion depends on the choice.

**Short answer.**

- No widefield stroke paper I could find applies a frozen decoder across the stroke, and none discusses this
  circularity or trial-mix problem.
- The closest relevant work comes from three areas: BCI decoder stabilization, the 2-photon drift literature, and
  ML label-shift theory.
- The BCI literature is the only one that states the circularity explicitly: mean subtraction can cancel real
  modulation (Jarosiewicz et al. 2015).
- That literature validates its methods by restored performance, which is the opposite of what we need. A method
  that restores decoding performance is exactly one that can hide a real change.

---

## 2. Approaches in the literature

### 2.1 Longitudinal / chronic widefield imaging, including stroke

#### What standard widefield pipelines do

Every pipeline I read normalizes each recording by its own F0. None applies an across-session z or a pooled frame.

| study | F0 / normalization | across-session handling | sourcing |
|---|---|---|---|
| Allen et al. 2017, *Neuron* | per-pixel ΔF/F with F = median of the entire time series; scaled 410 nm ΔF/F subtracted from 488 nm ΔF/F | atlas registration (CCFv2, affine, four control points) | full text |
| Peters et al. 2021, *Nature* (striatum + widefield) | violet-scaled hemodynamic subtraction; linear detrend; 0.01 Hz high-pass; ΔF/F0 = divide by each pixel's mean fluorescence "softened by adding the median average fluorescence across pixels" | days aligned by rigid registration of each day's mean violet (vasculature) image; mice aligned by retinotopy | full text |
| Couto et al. 2021, *Nat Protoc* (wfield) | F0 = "average of a number of frames before trial onset", or the mean of all session frames if that is not possible | chronic imaging over months; Fig. 1e shows two recordings 6 weeks apart in which task activity "strongly increases with learning", each with its own per-session dF/F | full text |
| Musall et al. 2019, *Nat Neurosci* | baseline-corrected ΔF/F; CCF alignment; SVD; encoding models fit within session | learning analyzed as per-session cvR² over training days, not with a cross-session decoder | full text |
| Pinto et al. 2019, *Neuron* | F0 = mode over a 30 s sliding window; divisive hemodynamic correction ΔF/F = R_b/R_v − 1 | — | full text |
| MacDowell & Buschman 2020, *Curr Biol* | F0 = rolling mean over a 9,750 ms window, then linear detrend | — | full text |
| Makino et al. 2017, *Neuron* (motor learning over weeks) | F0 = 10th percentile over 30 s around each time point | ROI Δf/f compared across learning; they report "no net change in the amplitude of movement-related activity" | full text |
| Murphy et al. 2020, *eLife* (home-cage, longitudinal) | dF/F0 with blue-reflectance hemodynamic subtraction | — | full text |
| Saxena et al. 2020, *PLoS CB* (LocaNMF) | DF/F after hemodynamic correction, SVD to 200 components | SVD components are "typically not reproducible across recording sessions", hence atlas-constrained LocaNMF. Spatial maps are similar across sessions (their Fig. 6), but each session is decomposed separately | full text |

So per-recording ΔF/F0 is the universal first step. *Our reading:*

- Per-session F0 is itself a per-session divisive normalization.
- Any post-stroke change in F0 or absorption already enters every frame we test, including N, unless N is built
  on raw rather than ΔF/F units. Such changes include autofluorescence, expression, and the light-absorption change
  over the lesion that Bandet & Winship 2024 show.

I found no widefield paper that trains a decoder on some sessions and applies it frozen to others. Within-session
cross-validation is the norm.

#### Stroke widefield / mesoscale studies

| study | modality, model | normalization | what it shows for us | sourcing |
|---|---|---|---|---|
| Lim, LeDue, Mohajerani & Murphy 2014, *J Neurosci* | ChR2 point stimulation + VSD imaging; cortical stroke; 1 and 8 weeks | — | **Normalization can hide a global effect.** At 1 week evoked activity is widely depressed, including in the non-injured hemisphere. "scaling the ChR2-evoked VSD responses from the stroke groups to match the sham group mean resulted in a relative distribution of responses that was indistinguishable from the sham group". A global down-scaling ("connectional diaschisis") disappears under a global rescale. A per-session z would do the same. | abstract only |
| Bauer et al. 2014, *NeuroImage* (Lee / Culver labs) | OIS functional connectivity, tMCAO | GSR vs multiple-signal regression (MSR: separate infarct and non-infarct regressors) | **The most direct "conclusion depends on normalization" result in stroke imaging.** "With increasing stroke severity, homotopic functional connectivity switches sign using global regression and increases, but decreases using multiple signal regression." GSR alone "resulted in apparently distorted functional connectivity measures in the intact hemisphere", and "it is necessary to properly account for altered hemodynamics in injured brain". | full text |
| Bice et al. 2022, *eLife* | OIS + optogenetics, photothrombosis S1FP, 4 weeks | GSR before RSFC and evoked analyses; affine to atlas; magnitude and area "normalized by the mean baseline values" | pre-stroke baseline as the reference for each animal | full text |
| Cramer et al. 2019, *NeuroImage* (Liesz lab) | widefield GCaMP, photothrombotic motor-cortex stroke, to day 56 | per-recording ΔF/F with F = mean of the time course; 0.1–1 Hz band-pass; connectivity "normalized to baseline" | autofluorescent (saturated) pixels from necrotic tissue rise sharply after stroke and fall during recovery. They are masked, and ROIs touching them are excluded. This is a post-stroke change in F that a per-session F0 absorbs. | full text (PDF) |
| Vignozzi et al. 2025, *PLoS Biol* | widefield GCaMP7f in PV interneurons, stroke + rehabilitation, D2–D28 | ΔF/F with F = mean over time; ratiometric hemodynamic correction; GSR | **A robustness check:** FC re-run with and without GSR. "GSR shifted the overall FC values toward lower correlations ... However, global network alterations after stroke were consistent" (S4–S6 Figs). | full text |
| Balbi et al. 2019, *JCBFM* (Murphy lab) | awake widefield GCaMP6, microinfarcts, longitudinal | — | graph-theoretic FC showed no sham vs microinfarct difference despite behavioral deficits | abstract only |
| Bandet & Winship 2024, *eLife* | 2-photon + widefield, photothrombosis, 2 months | ΔF/F0 per session | peak forelimb ΔF/F0 falls at 8 weeks; Fig. 1D shows "the area of decreased light absorption due to stroke damage", which shrinks over time | full text |
| Lee, Jadav & Landsness 2026, preprint | widefield Thy1-GCaMP6f, photothrombosis | — | **Rest itself changes after stroke.** Ipsilateral slow-oscillation (0.1–1 Hz) power is acutely suppressed and partly recovers by 1 week | abstract only (preprint) |

**Gaps.**

- None of these studies decodes task variables across the stroke. All compare per-session summary measures: maps,
  FC, evoked amplitude.
- None discusses the trial-mix or label-shift problem, because none has a multi-class behavioral readout whose
  class mix changes post-stroke.
- I did not identify relevant widefield-stroke papers from Silasi, Carmichael (Latifi et al. 2020 is 2-photon,
  abstract only), Wright/Lee (beyond Bauer 2014 / Bice 2022 / Wright 2017) or a "Bhatt / Kim" group. I also did not
  find a Chen / Hires / Svoboda widefield paper relevant to cross-session frames. These names are not covered here.
- Steinmetz et al. 2019 (*Nature*) is Neuropixels, not widefield. It is not relevant to the frame question.

### 2.2 Decoders across days with drifting neural data (2-photon drift, latent alignment)

**Driscoll et al. 2017, *Cell*** [full text]

- Posterior parietal cortex, 2-photon, a month of a stable virtual-navigation task.
- Decoders trained on one day lose accuracy on other days, while the population keeps "statistically similar
  properties each day".
- No re-centering. This is the motivating phenomenon.

**Rule et al. 2020, *eLife*** [full text]

- Driscoll's data. Drift is "systematically constrained", but "a significant component of drift continually degrades
  a fixed readout, implying that drift is not confined to a null coding space."
- They compare:
  - single-day fixed decoders;
  - "concatenated decoders" fit to several days;
  - an online LMS rule on "an affine decoder (i.e. a linear decoder with a constant offset feature or bias
    parameter)", which tracks drift with 7.6–10.4 % weight change per day.
- Their drift-vs-noise analysis z-scores log fluorescence and conditions trial means on location and cue.
- Rule, O'Leary & Harvey 2019 (*Curr Opin Neurobiol*) [abstract only] is the review.

**Ziv et al. 2013** (*Nat Neurosci*), **Deitch, Rubin & Ziv 2021** (*Curr Biol*) [abstract only]

- Drift in CA1 and visual cortex.
- Deitch: "the structure of the relationships between population activity patterns remained stable".
- *Our reading:* this supports readouts based on relational structure (correlation or geometry), which are
  insensitive to per-session offset and gain.

**Marks & Goard 2021, *Nat Commun*** [full text]

- Their drift metric, the RDI, compares within-session and between-session trial-to-trial correlation.
- Correlation is insensitive to per-session offset and gain, so the metric avoids the frame choice altogether.

**Gallego et al. 2020, *Nat Neurosci*** [full text, author manuscript]

- Method:
  - CCA aligns each day's latent dynamics (PCA of smoothed firing) to a reference day.
  - Trials are subselected to "the same number of trials for all sessions and targets", concatenated, and "ordered
    by time and target". CCA therefore pairs data by condition and time, so it uses labels.
  - The aligned decoder is trained on day A latents and tested on day B latents after alignment.
  - The classifier training set was balanced by target ("the same number of trials for each target").
- Validation:
  - an upper bound from within-day split halves, "matched by target and time points";
  - a control in which surrogate data are built from each day's tuning curves. Its CCs were lower than real
    across-day CCs, so alignment does not trivially make any two datasets look alike.
- Limits:
  - the claim is stability under "consistent behavior";
  - behavior was verified to be matched across days (hand-velocity correlations);
  - if latent dynamics "were to change fundamentally ... no linear transformation would be able to compensate".
- *Our reading:* CCA fit on all conditions would absorb a real condition-specific post-stroke change. The design
  only works for a stability claim.

**Degenhart et al. 2020, *Nat Biomed Eng*** [full text, author manuscript]

- Factor analysis is refit on recent data. Its parameters Λ (loadings), **μ (per-channel mean)** and Ψ are
  "successively re-estimated".
- The new loadings are rotated onto the baseline loadings by orthogonal Procrustes, but only over **"stable
  electrodes"**: electrodes whose covariance with the others is unchanged.
- So the stabilizer does mean re-estimation plus a rotation anchored on the parts of the data assumed unchanged.
- It "does not require knowledge of user intent".
- Validation: online monkey BCI with artificially induced instabilities (electrode dropout, swaps), single-day and
  multi-day. It recovered control and could outperform supervised recalibration.
- Stated limits:
  - "any instabilities that do not result in a change in covariance will be missed";
  - it needs more stable electrodes than latent dimensions;
  - too few assumed dimensions "severely" hurts.
- *Our reading:* the stable-electrode anchor is the closest published analog of our reference-condition anchor.
  The difference is that the anchor is a subset of channels, not of task conditions.

**Farshchian et al. 2018 (ADAN)** [abstract only, arXiv]; **Ma et al. 2023, *eLife*** [full text]

- These use adversarial (ADAN) or Cycle-GAN distribution alignment of latent or full-dimensional activity, compared
  with "Procrustes Alignment of Factors" (Degenhart).
- Farshchian's abstract: CCA "requires matched time points", whereas KL-based alignment does not.
- Ma et al. on CCA: it requires that "paired trials correspond to the same behavior and contain the same number of
  timepoints". Without trial alignment, "no amount of spatial rotation will achieve a correlation".
- Distribution alignment does not need labels. *Our reading:* it does implicitly need the behavior distribution to
  be the same on both days, which is the label-shift assumption in another form.

**Karpowicz et al. 2025, *Nat Commun* (NoMAD)** [full text]

- An LFADS dynamics model is frozen. An alignment network maps day-K data so that the generator-state distribution
  matches day 0. The match is a KL divergence between Gaussian approximations, so first and second moments only.
- Stated limitation: manifold alignment relies "on a stable relationship between activity on the neural manifold and
  behavior". This may fail with learning.
- It needs "similar enough behavioral distributions".
- They note that Dyer et al. 2017 distribution alignment "fails when distributions of movements are symmetric".

**Dyer et al. 2017, *Nat Biomed Eng*** [abstract only]

- Decoding by aligning decoder outputs to the expected distribution of movements (KL).
- *Our reading:* a label-prior-matching approach. It assumes the behavior distribution is known.

**What 2.2 shows.**

- (a) Re-centering or mean re-estimation is a standard part of these stabilizers (Degenhart μ; NoMAD first moments).
- (b) Latent alignment (CCA / Procrustes / KL) is the main tool. Every variant assumes the same behavior and the
  same neural-to-behavior mapping, and is validated by restored decoding.
- (c) Intercept or bias recalibration appears as the offset term in Rule's affine LMS decoder, and as a separate
  stage in the BCI papers (2.3).
- (d) Label-shift EM does not appear in any of these papers.
- None asks whether alignment absorbs a real change, because their goal is to remove change. Gallego's surrogate
  control is the nearest thing to a check that alignment is not trivially successful.
- Class imbalance is handled by subsampling to equal trials per target (Gallego; Degenhart: "Trials were removed
  from the calibration set to preserve a uniform number of trials across target conditions"). *Our reading:* this
  is the analog of our Wt.

### 2.3 Intracortical BCI recalibration: fixed weights + mean / bias re-estimation

**Jarosiewicz et al. 2015, *Sci Transl Med*** [full text, author manuscript]

Three components:

- **mean tracking:** feature means re-estimated during self-timed *pauses* in control and frozen at the start of
  the next block;
- **velocity bias correction:** an exponentially weighted running mean of decoded velocity, subtracted from the
  output;
- **retrospective target inference** recalibration.

They state the circularity directly. If mean estimation runs during control with a short time constant, "mean
subtraction can dampen the effects of actual neural modulation related to voluntary movement intent and cause a
subsequent bias opposite the intended movement". They also describe a feedback loop: "the neural activity resulting
from counteracting the bias would enter into the estimation of the new baseline rates".

Their fix was to estimate means only from pauses, so the estimate does not depend on task demands. That is the BCI
analog of our R5 / RW rest center. They correct the remaining bias at the output, which is the analog of an
intercept-only refit.

- Validation: hours to days of self-paced typing. With all methods off, typing rate declined (r = −0.85, −0.87), and
  reinstating them rescued control.
- *Our reading:* this validates "performance is maintained", not "real changes are preserved". Pauses in a BCI user
  are also assumed to be state-neutral. After a stroke our rest is not neutral (2.5).

**Wilson et al. 2026, *Nat Biomed Eng*** [full text]

This is the most systematic comparison of recalibration methods. On a ~5-year dataset from one participant, a
decoder trained on an earlier session is tested on a later one after each of:

- supervised recalibration;
- **mean recalibration**;
- FA stabilization (Degenhart);
- ADAN;
- their HMM target-inference method (PRI-T).

Findings:

- They separate two kinds of nonstationarity:
  - "baseline shifts", which "maintain a channel's response specificity to direction but alter its mean activity";
  - preferred-direction (PD) changes.
- "mean shifts can easily be accounted for with adaptive offset-correction methods". A miscalibrated mean gave
  R² = −0.21 with correlated predictions, "a strong bias rather than a tuning mismatch". Re-estimating the mean
  raised R² to 0.23.
- "All methods generally outperformed mean recalibration", so mean recalibration is the floor.
- With iterative (chained) use, distribution-matching approaches "appear to accumulate compounding errors over time",
  and FA stabilization "appears to accumulate error until eventually diverging".

Relevance:

- The baseline-shift vs tuning-change split is exactly the split we need. A frame should remove baseline shift and
  keep tuning change.
- Mean recalibration is the field's minimum expected comparison.

**Sussillo et al. 2016, *Nat Commun*** [full text]

- No recalibration at all. They train on months of data, sampling "the same number of trials from each day" per
  minibatch.
- They also augment training data with "random Gaussian firing rate scaling ... robust to both global and private
  shifts in observed firing rates".
- The decoder became robust to recording changes that disabled a Kalman filter.
- This is the "make the frozen decoder frame-invariant" alternative. *Our reading:* it maps onto N plus training-time
  offset and gain augmentation.

**Other BCI work**

- **Dangi et al. 2013, *Neural Comput*** [abstract only]: closed-loop decoder adaptation. Design elements include
  "selective parameter adaptation", which means adapting only some parameters (for us, intercepts only).
- **Fan et al. 2023, NeurIPS (CORP)** [abstract only]: language-model pseudo-labels for continual recalibration.

**Validation against true changes.** None of the BCI papers validates recalibration against a known, preserved
neural change. All validate by restored performance. This is a real gap.

### 2.4 ML label shift vs covariate shift, and its use in neural decoding

**Saerens, Latinne & Decaestecker 2002, *Neural Comput*** [abstract only]

- EM re-estimates the new class priors from unlabelled test data and adjusts the posteriors without refitting.
- *Our reading:* for a softmax or LDA classifier, the posterior adjustment is the same as adding
  log(π̂_c / π_train,c) to each class intercept. Saerens is therefore an intercept-only adjustment at the output,
  not a feature-space centering.
- Our W uses the EM priors to build a class-balanced feature center. That is a hybrid. I found no published
  instance of EM-reweighted feature centering.

**Lipton, Wang & Smola 2018 (BBSE), ICML** [abstract only]

- Assumes the label marginal "changes but the conditional p(x|y) does not".
- Works with any black-box predictor "so long as their confusion matrices are invertible".
- Weaker predictors give looser estimates.

**Azizzadenesheli et al. 2019 (RLLS), ICLR** [abstract only]: a regularized weight estimator for small samples and
large shifts.

**Alexandari, Kundaje & Shrikumar 2020, ICML** [abstract only]

- Saerens' maximum likelihood "assumes ... p(y|x) is calibrated", which modern networks often are not.
- Maximum likelihood with bias-corrected calibration beats BBSL and RLLS.
- The objective is concave.

**Garg et al. 2020, NeurIPS** [abstract only]

- MLLS (Saerens' maximum likelihood) is consistent only with a calibrated classifier and an invertible confusion
  matrix.
- Its error splits into a miscalibration term and a finite-sample term.

**Failure modes that apply to us** (*our reading*, from the stated assumptions above):

- **Concept shift.** The stroke may change p(x|y) itself, for example a different far-contra pattern. That is the
  effect of interest, and it breaks the core assumption of all label-shift methods. The EM then attributes a
  conditional change to a prior change. In our CENTRE_MIXSHIFT doc this is the "(i) EM confuses spout priors"
  mechanism.
- **Weak classifier, near-singular confusion.** Adjacent spouts are confusable and post-stroke accuracy is low, so π̂
  is poorly identified.
- **Absent classes.** EM can drive π̂_c to 0. Mass belonging to an absent class is assigned to its confusable
  neighbors.
- **Calibration.** The frozen decoder's probabilities on a post-stroke session are probably miscalibrated, and
  Alexandari / Garg show MLLS depends on calibration.

**Use in neural decoding**

- I found no paper applying Saerens / BBSE / MLLS to invasive or imaging neural decoding.
- The only neural case is EEG: **Li, Kawanabe & Kobler 2025 (SPDIM), ICLR** [full text, arXiv HTML, first ~100k of
  ~153k characters read].
  - They show that per-domain Riemannian re-centering, the EEG standard (Zanini et al. 2018 [abstract only]; He &
    Wu 2020 Euclidean alignment [abstract only]), "recovers domain-invariant representations if there are no label
    shifts". It "hurt[s] generalization under label shifts".
  - In their derivation the per-domain mean carries a prior-dependent term.
  - Their fix keeps the source model fixed and learns one bias parameter per target domain by information
    maximization.
  - They report that "no method can perfectly compensate" the label shifts they introduced.
  - This is the first published statement of our trial-mix problem for per-session centering.
- He & Wu 2020 (*IEEE TNSRE*) "label alignment" [abstract only] needs "as few as one labeled sample from each class"
  of the target. It is a small labeled calibration set.

### 2.5 Rest / baseline-period normalization, and when rest changes

- **Running-percentile and pre-stimulus baselines** are standard (2.1 table: Makino 10th percentile / 30 s; Pinto
  mode / 30 s; MacDowell rolling mean; Couto pre-trial frames).
- **Neugornet et al. 2021, *Front Neurosci*** [abstract only]
  - Compares F0 methods on one dataset: neuron and astrocyte Ca2+ imaging in rats, which the abstract calls
    widefield imaging.
  - "the choice of the processing method has a profound impact on the interpretation".
  - Most dF/F0 methods failed to correct bleaching.
- **Alday 2019, *Psychophysiology*** [abstract only]
  - Argues that subtracting a baseline is "statistically unnecessary and even undesirable".
  - Recommends entering the baseline as a GLM covariate, so the data decide how much correction to apply. Full and
    no correction are special cases.
  - *Our reading:* this is a principled middle ground between N and a rest center.
- **Rest is state-dependent:**
  - Wright et al. 2017 (*PLoS One*) [full text]: calcium FC structure differs between ketamine/xylazine and awake.
  - Ma et al. 2016 (*PNAS*) [abstract only]: neurovascular coupling is fast in awake mice and slower, with extra
    < 0.04 Hz components, under urethane.
  - Shimaoka et al. 2018 (*Cell Rep*) [abstract only]: arousal (running, pupil) depolarizes or hyperpolarises
    cortex in a modality-specific way.
- **Rest changes after stroke:**
  - Lee et al. 2026 preprint: slow-oscillation power suppressed acutely;
  - Bice 2022, Cramer 2019, Bauer 2014: RSFC changes;
  - Cramer 2019: autofluorescence changes F;
  - Bauer 2014: altered hemodynamics in injured tissue distort global regressors.
- **Hemodynamic correction** (Valley et al. 2020, *J Neurophysiol* [abstract only]: a spatial regression model fit
  per pixel) is done per session. Post-stroke changes in vascular structure or coupling will change what it removes.
  I found no paper that tests the stability of hemodynamic correction across a stroke.
- **The BCI precedent** (Jarosiewicz) uses rest (pauses) as the reference precisely because rest is assumed not to
  depend on the task. No paper I found validates a rest anchor when rest itself is altered by injury.

### 2.6 Studies that compare several normalization choices (robustness)

**Stroke imaging**

- Bauer et al. 2014: GSR vs MSR flips the sign of the post-stroke homotopic-FC effect.
- Vignozzi et al. 2025: GSR on/off, conclusions unchanged.
- Lim et al. 2014: a global rescale makes a network-wide stroke deficit vanish [abstract only].

**BCI**

- Wilson et al. 2026: five recalibration methods on the same 5-year dataset. Mean recalibration is the floor, and
  chaining causes divergence.
- Ma et al. 2023: three aligners (ADAN, Cycle-GAN, PAF) on the same data.
- Degenhart 2020 and Karpowicz 2025 compare against static and supervised decoders.

**General**

- Steegen et al. 2016 (*Perspect Psychol Sci*) [abstract only]: the "multiverse analysis". Run all reasonable
  processing choices and report how much the conclusion changes.
- Botvinik-Nezer et al. 2020 (*Nature*) [abstract only]: 70 teams analyzing one fMRI dataset gave "sizeable
  variation in the results of hypothesis tests".
- Murphy & Fox 2017 (*NeuroImage*) [abstract only]: GSR consensus. There is "not a single 'right' way", and different
  approaches "likely reveal complementary insights".
- Murphy et al. 2009 (*NeuroImage*) [abstract only]: GSR forces seed correlations to sum negative and can introduce
  anticorrelations.

I found no study that compares cross-session frames for a frozen decoder across an injury, and none that reports a
"robustness bracket" for that case. Our CENTRE_MIXSHIFT and REST_CENTRE analyses have no direct precedent.

---

## 3. Standard practice for widefield specifically

1. Per-recording ΔF/F0, with F0 from one of:
   - the session mean or median;
   - pre-trial frames;
   - a running mean, mode or low percentile over ~10–30 s.
2. Hemodynamic correction (violet/isosbestic or reflectance regression) per session.
3. Registration across days by vasculature (Peters), or to the Allen CCF by landmarks (Allen, Musall, Couto,
   LocaNMF).
4. Decomposition (SVD / LocaNMF) per session. Spatial components are not identical across sessions, which is why
   LocaNMF uses atlas constraints.
5. GSR or multi-signal regression in FC work. It is debated, and Bauer shows that after stroke the choice can
   reverse the sign of the effect.
6. Cross-session comparison of summary measures (maps, FC, evoked ΔF/F amplitude, within-session cvR²), normalized
   per session. Stroke papers express post values relative to each animal's pre-stroke baseline and include sham
   groups (Bice, Cramer, Bandet).

**There is no widefield standard for a frozen cross-session decoder frame.**

- The nearest widefield default a reviewer would recognize is per-session ΔF/F0 plus per-session z-scoring of
  features before decoding.
- The nearest neural-decoding default is mean recalibration (BCI).
- Reviewers from the BCI or drift side will know Gallego / Degenhart alignment and expect it to be considered.

---

## 4. Methods we are not testing that look worth trying

Ordered by how directly they address the circularity.

**1. Injected-effect validation of every frame.** This is the most important one, and it has no published
precedent, which is a gap in the literature.

- Take held-out pre sessions.
- Inject a known, spout-specific change of realistic size into the features. For example, shrink or rotate the
  far-contra class mean by a set fraction, or add a global gain change as in Lim 2014.
- Combine it with the observed post-stroke trial-mix shift (the existing mix-shift resampler).
- Score the change under each frame. The best frame keeps the injected effect and has no effect under the
  mix-shift-only control.
- This is the "validate against a true change" test that the BCI literature never ran. It extends our Test 2 with a
  known signal on top of the mix shift.

**2. Output-side prior correction with the TRUE trial mix (Saerens with known priors).**

- We know each post session's spout counts from the DAQ, so the label shift does not need to be estimated.
- Use N (or Wt) as the feature frame. Correct the frozen decoder's posteriors, equivalently its intercepts, by
  log(π_true,c / π_train,c).
- This separates "the trial mix changed" (handled at the output with known numbers) from "the pattern changed" (left
  in the features).
- It avoids the concept-shift failure of EM, because nothing is estimated from the features. It applies only to
  probabilistic or accuracy readouts, not to the C projections.
- Our intercept-only refit learns intercepts from post data. This variant fixes them from known priors, with no fit.

**3. Reference-condition alignment (CCA or Procrustes on stroke-unaffected conditions), à la Gallego / Degenhart.**

- Fit an affine or orthogonal map from post to pre latent space using only the reference spouts' trials, matched by
  spout and time bin as in Gallego.
- Apply the map to all spouts. Test whether the held-out affected spouts are still decoded.
- Degenhart's stable-electrode idea maps onto stable conditions.
- Required controls:
  - a pre-vs-pre alignment, which sets the noise ceiling (Gallego's within-day split);
  - surrogate data showing that the alignment cannot make an arbitrary change look stable (Gallego's tuning-curve
    surrogate).
- Risks:
  - Rotation can absorb a change shared with the reference spouts. A mean-only anchor (our reference anchor) cannot
    absorb anything beyond a shift.
  - If the "unaffected" spouts are partly affected, the anchor removes part of the effect.

**4. Calibrated label-shift estimation, as a diagnostic.**

- If W stays a candidate, compare against two calibrated estimates of π̂:
  - BBSE using the pre-stroke LOSO confusion matrix;
  - MLLS with bias-corrected calibration (Alexandari 2020), using held-out pre sessions as the calibration set.
- Report π̂ against the true mix for every post session. We have labels, so the label-shift estimate can be checked
  directly, which ML papers usually cannot do.
- Large π̂ errors at far contra would show that the concept-shift failure is happening.

**5. Training-time invariance (Sussillo 2016).**

- Train the pre-stroke decoder with per-session random offset and gain jitter, and with equal trials per session and
  per spout.
- The frozen decoder then tolerates session-level offset and gain without any per-session center, so N becomes
  viable.
- Built-in check: the jitter must not be large enough to remove spout information.

**6. Baseline as a covariate, not a subtraction (Alday 2019).**

- In the C projections or encoding readouts, include the session's rest level (R5) as a covariate, with its slope fit
  on pre sessions.
- The data then decide how much rest correction to apply. This brackets N (slope 0) and full rest centering
  (slope 1).

**7. Offset- and gain-invariant readouts.**

- Report pattern correlation or a cross-validated distance between pre templates and post class means, as in Marks &
  Goard's RDI or Gallego's CCs.
- Per-session offset and scale cancel by construction.
- These cannot detect a pure global gain loss, which is the Lim 2014 effect. They complement N; they do not replace
  it.

**8. Sham or time-matched drift control.**

- Stroke widefield papers use sham groups (Bice, Cramer, Bandet).
- Without sham, the time-matched estimate is pre-to-pre drift over the same lag.
- Cohort 2 stays separate per lab rule. It is not a sham and is not pooled.

---

## 5. Recommendations for reporting

**Primary frame.**

- The decision rule: choose the frame that (i) uses no post-stroke labels or features that could carry the effect,
  and (ii) passes the injected-effect test (§4.1).
- Until that test is run, the defensible pairing is:
  - **N (pooled pre frame, no session center)** as the conservative frame. It cannot absorb a post-stroke change, but
    it carries every nuisance offset into the score.
  - **Wt (true-label balanced center)** as the mix-corrected frame. It removes the trial-mix confound with known
    labels, but absorbs any effect shared equally by all spouts.
- W is the label-free deployable version of Wt. The literature (SPDIM; label-shift theory under concept shift) gives
  specific reasons to report it alongside Wt, not instead of it.
- Rest frames (R5 / RW) are supplementary. The precedent (Jarosiewicz pauses) assumes rest is state-neutral, and
  after stroke it is not (Lee 2026; Cramer 2019; our own quiet-period event rates).

**Bracket, not a single number.**

- Report the key per-spout and per-epoch effect under N, Wt, W, Z, rest and the reference anchor, in one table or
  forest plot. This is a small multiverse (Steegen 2016).
- Say plainly which conclusions hold in every frame.
- State the direction of each frame's bias:
  - N can only overstate loss, because nuisance offsets lower accuracy;
  - Wt and Z can only understate a spout-general loss;
  - W can be biased in either direction under concept shift.
- If the far-contra deficit survives the frame that biases against it, say so. That is the strongest wording
  available.
- Always show held-out pre sessions (LOSO) under the same frame as the reference. Every published cross-day decoder
  normalizes by a within-day or pre benchmark (Gallego's normalized accuracy; Wilson's normalization to supervised
  recalibration).

**What to write in methods** (draft):

> Each session's features were expressed in the frozen decoder's frame by [primary]. Because any per-session
> normalization estimated from post-stroke data can absorb genuine post-stroke change (cf. Jarosiewicz et al. 2015)
> and per-session centering is biased by changes in the class mix (Li et al. 2025; Saerens et al. 2002), we report
> every effect under [N, Wt, W, rest, reference anchor], and we tested each frame on held-out pre-stroke sessions
> with the observed post-stroke trial mix imposed by resampling [and with a known injected change]. Held-out
> pre-stroke sessions scored with leave-one-session-out models provide the reference under each frame.

**What reviewers will ask:**

1. Does the center use the scored trials? Is it estimated on the same rows it is applied to, and could it absorb the
   effect? This is Jarosiewicz's point.
2. Does the post-stroke change in trial mix or failure rate move the center? Show the mix-shift control.
3. Is the result a global gain or offset change that z-scoring hides (Lim 2014), or a condition-specific change
   that z-scoring keeps? Show N.
4. Did F0, autofluorescence, absorption or hemodynamic correction change after the stroke? Show raw F0 maps over
   time, as in Cramer's autofluorescence masking.
5. Why not CCA / Procrustes alignment (Gallego, Degenhart)? Answer: alignment fit on all conditions removes the
   effect by design. Either show reference-condition alignment with held-out conditions, or explain why it was not
   run.
6. How does pre-to-pre drift over the same lag compare (Driscoll, Rule)? There is no sham, so use time-matched pre
   pairs.
7. Is the result driven by one animal or one frame? Show sessions as markers, per animal, under each frame.

---

## 6. References

Sourcing tags: **F** = full text read; **A** = abstract/metadata only; **P** = preprint.

**Widefield methods and practice**

- Allen WE, Kauvar IV, Chen MZ, Richman EB, Yang SJ, Chan K, Gradinaru V, Deverman BE, Luo L, Deisseroth K (2017).
  Global representations of goal-directed behavior in distinct cell types of mouse neocortex. *Neuron* 94:891–907.
  doi:10.1016/j.neuron.2017.04.017 [F]
- Couto J, Musall S, Sun XR, Khanal A, Gluf S, Saxena S, et al., Churchland AK (2021). Chronic, cortex-wide imaging
  of specific cell populations during behavior. *Nat Protoc* 16:3241–3263. doi:10.1038/s41596-021-00527-z [F]
- MacDowell CJ, Buschman TJ (2020). Low-dimensional spatiotemporal dynamics underlie cortex-wide neural activity.
  *Curr Biol* 30:2665–2680. doi:10.1016/j.cub.2020.04.090 [F]
- Makino H, Ren C, Liu H, Kim AN, Kondapaneni N, Liu X, Kuzum D, Komiyama T (2017). Transformation of cortex-wide
  emergent properties during motor learning. *Neuron* 94:880–890. doi:10.1016/j.neuron.2017.04.015 [F]
- Murphy TH, Michelson NJ, Boyd JD, et al., LeDue JM (2020). Automated task training and longitudinal monitoring of
  mouse mesoscale cortical circuits using home cages. *eLife* 9:e55964. doi:10.7554/eLife.55964 [F]
- Musall S, Kaufman MT, Juavinett AL, Gluf S, Churchland AK (2019). Single-trial neural dynamics are dominated by
  richly varied movements. *Nat Neurosci* 22:1677–1686. doi:10.1038/s41593-019-0502-4 [F]
- Peters AJ, Fabre JMJ, Steinmetz NA, Harris KD, Carandini M (2021). Striatal activity topographically reflects
  cortical activity. *Nature* 591:420–425. doi:10.1038/s41586-020-03166-8 [F]
- Pinto L, Rajan K, DePasquale B, Thiberge SY, Tank DW, Brody CD (2019). Task-dependent changes in the large-scale
  dynamics and necessity of cortical regions. *Neuron* 104:810–824. doi:10.1016/j.neuron.2019.08.025 [F]
- Saxena S, Kinsella I, Musall S, et al., Paninski L (2020). Localized semi-nonnegative matrix factorization
  (LocaNMF) of widefield calcium imaging data. *PLoS Comput Biol* 16:e1007791. doi:10.1371/journal.pcbi.1007791 [F]
- Valley MT, Moore MG, Zhuang J, et al., Waters J (2020). Separation of hemodynamic signals from GCaMP fluorescence
  measured with wide-field imaging. *J Neurophysiol* 123:356–366. doi:10.1152/jn.00304.2019 [A]
- Ma Y, Shaik MA, Kozberg MG, Kim SH, Portes JP, Timerman D, Hillman EMC (2016). Resting-state hemodynamics are
  spatiotemporally coupled to synchronized and symmetric neural activity in excitatory neurons. *PNAS*
  113:E8463–E8471. doi:10.1073/pnas.1525369113 [A]
- Wright PW, Brier LM, Bauer AQ, et al., Culver JP (2017). Functional connectivity structure of cortical calcium
  dynamics in anesthetized and awake mice. *PLoS One* 12:e0185759. doi:10.1371/journal.pone.0185759 [F]
- Shimaoka D, Harris KD, Carandini M (2018). Effects of arousal on mouse sensory cortex depend on modality. *Cell Rep*
  22:3160–3167. doi:10.1016/j.celrep.2018.02.092 [A]
- Neugornet A, O'Donovan B, Ortinski PI (2021). Comparative effects of event detection methods on the analysis and
  interpretation of Ca2+ imaging data. *Front Neurosci* 15:620869. doi:10.3389/fnins.2021.620869 [A]

**Stroke imaging**

- Balbi M, Vanni MP, Vega MJ, Silasi G, Sekino Y, Boyd JD, LeDue JM, Murphy TH (2019). Longitudinal monitoring of
  mesoscopic cortical activity in a mouse model of microinfarcts reveals dissociations with behavioral and motor
  function. *J Cereb Blood Flow Metab* 39:1486–1500. doi:10.1177/0271678X18763428 [A]
- Bandet MV, Winship IR (2024). Aberrant cortical activity, functional connectivity, and neural assembly
  architecture after photothrombotic stroke in mice. *eLife* 12:RP90080. doi:10.7554/eLife.90080 [F]
- Bauer AQ, Kraft AW, Wright PW, Snyder AZ, Lee JM, Culver JP (2014). Optical imaging of disrupted functional
  connectivity following ischemic stroke in mice. *NeuroImage* 99:388–401. doi:10.1016/j.neuroimage.2014.05.051 [F]
- Bice AR, Xiao Q, Kong J, et al., Lee JM, Culver JP, Bauer AQ (2022). Homotopic contralesional excitation suppresses
  spontaneous circuit repair and global network reconnections following ischemic stroke. *eLife* 11:e68852.
  doi:10.7554/eLife.68852 [F]
- Cramer JV, Gesierich B, Roth S, Dichgans M, Düring M, Liesz A (2019). In vivo widefield calcium imaging of the
  mouse cortex for analysis of network connectivity in health and brain disease. *NeuroImage* 199:570–584.
  doi:10.1016/j.neuroimage.2019.06.014 [F, PDF]
- Latifi S, Mitchell S, Habibey R, et al., Carmichael ST (2020). Neuronal network topology indicates distinct
  recovery processes after stroke. *Cereb Cortex* 30:6363–6375. doi:10.1093/cercor/bhaa191 [A]
- Lee J, Jadav AA, Landsness EC (2026). Slow oscillations track acute stroke injury but not functional recovery.
  Preprint, doi:10.64898/2026.06.18.732981 (DOI as listed by Europe PMC) [A, P]
- Lim DH, LeDue JM, Mohajerani MH, Murphy TH (2014). Optogenetic mapping after stroke reveals network-wide scaling of
  functional connections and heterogeneous recovery of the peri-infarct. *J Neurosci* 34:16455–16466.
  doi:10.1523/JNEUROSCI.3384-14.2014 [A]
- Vignozzi L, Macchi F, Montagni E, et al., Allegra Mascaro AL, Micera S, Caleo M, Spalletti C (2025). Combining
  gamma neuromodulation and robotic rehabilitation after a stroke restores parvalbumin interneuron dynamics and
  improves motor recovery in mice. *PLoS Biol* 23:e3002806. doi:10.1371/journal.pbio.3002806 [F]

**Drift and latent alignment**

- Deitch D, Rubin A, Ziv Y (2021). Representational drift in the mouse visual cortex. *Curr Biol* 31:4327–4339.
  doi:10.1016/j.cub.2021.07.062 [A]
- Degenhart AD, Bishop WE, Oby ER, Tyler-Kabara EC, Chase SM, Batista AP, Yu BM (2020). Stabilization of a
  brain-computer interface via the alignment of low-dimensional spaces of neural activity. *Nat Biomed Eng*
  4:672–685. doi:10.1038/s41551-020-0542-9 [F]
- Driscoll LN, Pettit NL, Minderer M, Chettih SN, Harvey CD (2017). Dynamic reorganization of neuronal activity
  patterns in parietal cortex. *Cell* 170:986–999. doi:10.1016/j.cell.2017.07.021 [F]
- Dyer EL, Gheshlaghi Azar M, Perich MG, Fernandes HL, Naufel S, Miller LE, Körding KP (2017). A cryptography-based
  approach for movement decoding. *Nat Biomed Eng* 1:967–976. doi:10.1038/s41551-017-0169-7 [A]
- Farshchian A, Gallego JA, Cohen JP, Bengio Y, Miller LE, Solla SA (2018). Adversarial domain adaptation for stable
  brain-machine interfaces. arXiv:1810.00045 (ICLR 2019). doi:10.48550/arXiv.1810.00045 [A]
- Gallego JA, Perich MG, Chowdhury RH, Solla SA, Miller LE (2020). Long-term stability of cortical population
  dynamics underlying consistent behavior. *Nat Neurosci* 23:260–270. doi:10.1038/s41593-019-0555-4 [F]
- Karpowicz BM, Ali YH, Wimalasena LN, et al., Miller LE, Pandarinath C (2025). Stabilizing brain-computer interfaces
  through alignment of latent dynamics. *Nat Commun* 16:4662. doi:10.1038/s41467-025-59652-y [F]
- Ma X, Rizzoglio F, Bodkin KL, Perreault E, Miller LE, Kennedy A (2023). Using adversarial networks to extend brain
  computer interface decoding accuracy over time. *eLife* 12:e84296. doi:10.7554/eLife.84296 [F]
- Marks TD, Goard MJ (2021). Stimulus-dependent representational drift in primary visual cortex. *Nat Commun*
  12:5169. doi:10.1038/s41467-021-25436-3 [F]
- Rule ME, O'Leary T, Harvey CD (2019). Causes and consequences of representational drift. *Curr Opin Neurobiol*
  58:141–147. doi:10.1016/j.conb.2019.08.005 [A]
- Rule ME, Loback AR, Raman DV, Driscoll LN, Harvey CD, O'Leary T (2020). Stable task information from an unstable
  neural population. *eLife* 9:e51121. doi:10.7554/eLife.51121 [F]
- Ziv Y, Burns LD, Cocker ED, et al., Schnitzer MJ (2013). Long-term dynamics of CA1 hippocampal place codes. *Nat
  Neurosci* 16:264–266. doi:10.1038/nn.3329 [A]

**BCI recalibration**

- Dangi S, Orsborn AL, Moorman HG, Carmena JM (2013). Design and analysis of closed-loop decoder adaptation
  algorithms for brain-machine interfaces. *Neural Comput* 25:1693–1731. doi:10.1162/NECO_a_00460 [A]
- Fan C, Hahn N, Kamdar F, et al., Willett FR (2023). Plug-and-play stability for intracortical brain-computer
  interfaces: a one-year demonstration of seamless brain-to-text communication. *NeurIPS* 36:42258–42270. PMCID
  PMC11086983 [A]
- Jarosiewicz B, Sarma AA, Bacher D, et al., Hochberg LR (2015). Virtual typing by people with tetraplegia using a
  self-calibrating intracortical brain-computer interface. *Sci Transl Med* 7:313ra179.
  doi:10.1126/scitranslmed.aac7328 [F]
- Sussillo D, Stavisky SD, Kao JC, Ryu SI, Shenoy KV (2016). Making brain-machine interfaces robust to future neural
  variability. *Nat Commun* 7:13749. doi:10.1038/ncomms13749 [F]
- Wilson GH, Stein EA, Kamdar F, et al., Henderson JM, Willett FR (2026). Long-term unsupervised recalibration of
  cursor-based intracortical brain-computer interfaces using a hidden Markov model. *Nat Biomed Eng* 10:1466–1484.
  doi:10.1038/s41551-025-01536-z [F]

**Label shift / domain adaptation**

- Alexandari A, Kundaje A, Shrikumar A (2020). Maximum likelihood with bias-corrected calibration is hard-to-beat at
  label shift adaptation. *ICML 2020*. arXiv:1901.06852 [A]
- Azizzadenesheli K, Liu A, Yang F, Anandkumar A (2019). Regularized learning for domain adaptation under label
  shifts. *ICLR 2019*. doi:10.48550/arXiv.1903.09734 [A]
- Garg S, Wu Y, Balakrishnan S, Lipton ZC (2020). A unified view of label shift estimation. *NeurIPS 2020*.
  arXiv:2003.07554 [A]
- He H, Wu D (2020). Transfer learning for brain-computer interfaces: a Euclidean space data alignment approach.
  *IEEE Trans Biomed Eng* 67:399–410. doi:10.1109/TBME.2019.2913914 [A]
- He H, Wu D (2020). Different set domain adaptation for brain-computer interfaces: a label alignment approach. *IEEE
  Trans Neural Syst Rehabil Eng* 28:1091–1108. arXiv:1912.01166 [A]
- Li S, Kawanabe M, Kobler RJ (2025). SPDIM: source-free unsupervised conditional and label shift adaptation in EEG.
  *ICLR 2025*. arXiv:2411.07249 [F, partial]
- Lipton ZC, Wang Y-X, Smola A (2018). Detecting and correcting for label shift with black box predictors. *ICML
  2018*. arXiv:1802.03916 [A]
- Saerens M, Latinne P, Decaestecker C (2002). Adjusting the outputs of a classifier to new a priori probabilities:
  a simple procedure. *Neural Comput* 14:21–41. doi:10.1162/089976602753284446 [A]
- Zanini P, Congedo M, Jutten C, Said S, Berthoumieu Y (2018). Transfer learning: a Riemannian geometry framework
  with applications to brain-computer interfaces. *IEEE Trans Biomed Eng* 65:1107–1116.
  doi:10.1109/TBME.2017.2742541 [A]

**Robustness / analysis flexibility / baselines**

- Alday PM (2019). How much baseline correction do we need in ERP research? Extended GLM model can replace baseline
  correction while lifting its limits. *Psychophysiology* 56:e13451. doi:10.1111/psyp.13451 [A]
- Botvinik-Nezer R, Holzmeister F, Camerer CF, et al. (2020). Variability in the analysis of a single neuroimaging
  dataset by many teams. *Nature* 582:84–88. doi:10.1038/s41586-020-2314-9 [A]
- Murphy K, Birn RM, Handwerker DA, Jones TB, Bandettini PA (2009). The impact of global signal regression on resting
  state correlations: are anti-correlated networks introduced? *NeuroImage* 44:893–905.
  doi:10.1016/j.neuroimage.2008.09.036 [A]
- Murphy K, Fox MD (2017). Towards a consensus regarding global signal regression for resting state functional
  connectivity MRI. *NeuroImage* 154:169–173. doi:10.1016/j.neuroimage.2016.11.052 [A]
- Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W (2016). Increasing transparency through a multiverse analysis.
  *Perspect Psychol Sci* 11:702–712. doi:10.1177/1745691616658637 [A]

*Not cited because not verified:* the Jia et al. 2011 ΔF/F method (only its abstract was available, and it does
not describe the baseline method), and Chen / Hires / Svoboda widefield work (not located).

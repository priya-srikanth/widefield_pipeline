# How does the brain allow recovery after a VLS stroke? Literature in the context of our findings, and what to do next (2026-10-30)

Priya 10-30: "it seems like we can't really study coherence. Do a deep dive of the stroke recovery literature and systems
neuroscience of licking literature, in the context of our findings. What other analyses would be helpful in trying to
answer the biological question of how does the brain allow functional recovery after stroke?"

Three literature scans (stroke recovery mechanisms; licking circuits; analysis methods), each citation checked against a
page the helper saw (PubMed / PMC / journal); [unverified] = not fully checked. Our findings: `docs/FINDINGS.md`.
Companion: `docs/CROSS_AREA_METHODS_LIT_20261030.md` (cross-area methods; all failed or demoted here).

## 1. What the literature predicts for a LEFT VLS stroke, and where our results sit

| Literature | Prediction for us | Our result | Reading |
|---|---|---|---|
| VLS dSPNs drive contraversive licks via lateral SC (Lee, Wang & Sabatini 2020 Nat Neurosci); the two lateral SCs compete, iSPNs bias ipsiversive (Lee & Sabatini 2021 Nature) | losing left VLS -> right SC dominant -> licks biased ipsilesionally, worse the further contralesional the target | acute ipsilesional aim from launch, graded to far contra (FINDINGS 1-2) | **as predicted**: the deficit is a subcortical directional-drive loss |
| VLS D1 / D2 set lick bout structure and contact (Chen 2021 Cell Rep); striatal dopamine loss lowers tongue force / rhythm (Skitek 1999); BG scale vigour (Yttri & Dudman 2016) | vigour / rhythm changes, not only aim | acute ILI +12.5 ms, global slowing (FINDINGS 1; STATUS 09-20) | consistent |
| Learned skill kinematics live in the BG, cortex is needed to learn, not execute (Kawai 2015; Dhawale 2021) | recovery of the old kinematics needs BG; a new solution may need cortex | success returns, kinematics do not (compensation, FINDINGS 0 item 4b) | **compensation, as predicted when the BG store is damaged** |
| ALM is needed for online corrections of missed licks (Bollu 2021 Nature); ALM / tjM1 signal a missed target (Xu 2022 Nature); touch-guided re-aiming needs lateral SC, not cortex (Ito 2025 Nature) | if recovery runs on corrections, cortical lick responses grow with correction demand; or corrections may be collicular | lick-evoked responses LARGER bilaterally; far-contra code carried by later, corrective licks (FINDINGS 0b, 4d0b) | **open, and testable now (2.1)** |
| Unilateral ALM silencing is rescued by the other hemisphere (Li 2016 Nature); balance-recovery / vicariation models (Di Pino 2014 [partial]); contralesional excitation blocks repair (Bice 2022 eLife); 2025 preprint: contralesional activity = compensatory use, not recovery [unverified authors] | hemispheric balance shifts; may be compensation rather than mechanism | orofacial balance tracks how far each animal recovers (FINDINGS 0b) | **needs the kinematics control (2.4)** before it is a mechanism |
| Subcortical stroke in humans: bilateral over-activation that falls with recovery and is larger in worse patients (Ward 2003 Brain); diaschisis (Carrera & Tononi 2014 Brain) | larger responses early, shrinking with recovery; changes in cortex that projects to the lesion | residual excess largest day 1, shrinks with recovery, tracks the deficit (mostly PS94) (FINDINGS 0 item 3) | consistent; the corticostriatal-input map (2.5) is the subcortical-specific test |
| Remapping is mostly potentiation of existing circuits, not new territory (Makin & Krakauer 2023 eLife); remapped areas can be causally necessary (Castro-Alamancos & Borrell 1995) | the "remapped" code is re-weighting inside old territory | code partly remapped, not restored; position package shifts contralesionally (FINDINGS 0 items 3-4) | **re-embedding vs new pattern not yet tested (2.3)** |
| Capsular (subcortical) stroke: slower recovery, no lasting ipsilesional cortical reorganisation (2025 preprint [unverified authors]); cortex + striatum become more reliable / aligned with recovery (Guo 2021 Cell Rep) | little cortical reorganisation; recovery may be elsewhere | no evidence cortex leads behavioural recovery; chronic mapping change ~ cortex-wide, drift-like (FINDINGS 0 item 4e, 3) | consistent with **recovery largely outside dorsal cortex**; we cannot see it |

**Working model to test.** Losing left VLS removes the subcortical drive and the stored kinematics for contralesional
licks. Success returns within ~2 weeks through a COMPENSATORY strategy: a biased launch fixed by later licks and
corrections, with a changed jaw / tongue pattern. Dorsal cortex follows this strategy: larger bilateral lick responses,
a shifted balance, a re-weighted spout code. Whether cortex is NEEDED for the recovered behaviour, or only reports it, is
not decidable from imaging. That needs a perturbation (section 3).

## 2. Analyses on cohort 1 (ranked by value for the biological question)

1. **Is recovery carried by corrections? (Bollu 2021 / Xu 2022)** 3-D tongue at 250 fps resolves corrective
   submovements (Bollu used 1 kHz, Xu 400 Hz; check the effective 3-D rate first).
   - Behaviour: per lick, launch angle vs contact angle (correction = contact minus launch); corrective-submovement
     count (acceleration peaks); hypometria; whether the first lick misses and a later one corrects. Per spout and epoch.
     Prediction under compensation: launch bias persists while the correction share of hits rises, especially far contra.
   - Neural: per-lick lick-evoked amplitude in ALM / tjM1 / SSp-m on correction licks vs non-correction licks, with
     kinematics matched. Does the larger post-stroke lick response belong to correction licks? Does a missed first lick
     evoke a cortical miss / error response (Xu 2022) that grows post-stroke?
   - This is the most direct link between the behavioural compensation and the cortical changes we have.
2. **Formal restitution vs compensation score (SRRR; Levin 2009).** Per trial, distance of the kinematics from the
   animal's pre-stroke distribution at that spout (launch angle, jaw lateral steering gain, protrusion, speed,
   tongue-jaw lag, landing): Mahalanobis or a pre-vs-post classifier probability. Plot against hit rate per day; fit time
   constants per animal for hit rate, kinematic normality and each cortical readout. **Which cortical measures follow
   success (fast) and which follow kinematic normality (slow / incomplete)?** Hit rate vs aim half-times exist already
   (Fig 0.12); this adds the full kinematic signature.
3. **Re-embedded old dynamics vs new dynamics (Gallego 2020; Degenhart 2020; Makin & Krakauer 2023).**
   - (i) Frozen pre-stroke decoder; (ii) the same decoder after a low-dimensional linear alignment (CCA / Procrustes of
     k ~ 8-12 LocaNMF or PCA latents, fit on condition-averaged, kinematics-matched data, labels never used);
     (iii) full refit.
   - Index = (ii - i) / (iii - i), with a pre-to-pre null at matched session lags. Near 1: the old code is present but
     re-embedded (inspect the alignment matrix for which areas carry it). Near 0: a new code.
   - Pixel version: overlap / centroid shift of the spout-informative territory, pre vs chronic. Inside old territory =
     re-weighting, not vicariation.
   - Pitfall: alignment can map anything onto anything. Fix k in advance and cross-validate. Safaie 2023: different
     behaviour alone breaks alignment, so match kinematics first.
4. **Is the hemispheric balance a mechanism or a compensation marker?** Refit the balance after regressing per-lick
   3-D kinematics + FaceRhythm factors (the 2025 preprint's critique). Partly done: the balance survives vigour and
   landing regression for some readouts; finish it on the recovery-tracking readout.
   - If the kinematics absorb the balance-recovery link, report the balance as a cortical signature of the compensatory
     movement.
5. **Corticostriatal-input map (subcortical diaschisis; Carrera & Tononi 2014; Hunnicutt 2016 [unverified]).** Pixel-wise
   Allen connectivity projection density INTO the lesioned VLS volume (from our histology extent) as a spatial regressor
   of each post-stroke map change (response gain, decoder weight, residual). Prediction if the cortical changes come from
   losing the striatal target: change scales with projection density, beyond distance from orofacial cortex.
   - Cheap if the Allen projection data can be read without a pip install (HTTP API / downloaded volumes).
6. **Error / outcome signals and trial-to-trial adjustment (relearning; Kawai 2015).** Cortical response to a miss or
   no-reward on trial t vs the aim adjustment on trial t+1 at the same spout. Does this coupling strengthen in the
   subacute window, when success is climbing?
   - A learning-like signature would argue that cortex helps build the compensation, not only reports it. Control:
     slow within-session trends (pseudosession null).
7. **Unsupervised check: tensor component analysis across all sessions per animal** (Williams 2018). Area x
   time-in-trial x trial factors: a factor that drops and recovers vs one that appears new. Exploratory only; drift
   confounds slow factors.
8. **Control for S20 (information timing).** Re-align to jaw opening instead of tongue onset. If the motor-first lead
   disappears, it was movement timing, not cortical order.

**Stop / do not do:** phase, coherence, MVAR, LFADS-style dynamics (indicator kinetics); proportional-recovery or
baseline-vs-change correlations (Hope 2019 coupling artefact); any across-animal predictor claim (n = 4, rank order
only); naive cross-lagged session analyses (Hamaker 2015; Harris "nonsense correlations": use detrended residuals and
pseudosession nulls; our existing hits-only test found no lead either way).

**Statistics for all of the above:** session = unit; per-animal tables first; mixed models with an animal random
effect; within-animal controls (spared ipsilesional spouts = difference-in-differences; the contralesional hemisphere);
pre-stroke drift rate as the null for slow change.

## 3. Cohort 2 (PS118 / 120-122; imaged since 10-29, stroke not yet made): design questions for Priya

Cohort 1 cannot answer these by imaging alone; cohort 2 can if decided before the stroke.
- **Drift null:** a longer pre-stroke imaging period (several weeks) gives a time-matched no-stroke drift reference for
  every "remapping" claim. This is the main missing control in cohort 1.
- **Necessity of cortex in recovered animals** (Castro-Alamancos & Borrell 1995 logic; Bollu 2021; Kawai 2015). Brief
  inactivation (e.g. muscimol, or optogenetic where the line allows) of ipsilesional vs contralesional ALM / tjM1:
  - first pre-stroke, then in the chronic phase, scored on launch angle, corrections and hit rate;
  - if the chronic deficit returns only with cortex silenced, cortex carries the compensation;
  - Thy1-GCaMP6s and RiboL1 lines do not carry ChR2; pharmacology or a viral approach would be needed.
- **Histology for the subcortical account:** TH in SNc / striatum, VM / VAL thalamus and lateral SC gliosis or neuron
  loss, lesion extent mapped to the corticostriatal input zone; also on cohort 1 if tissue allows.
- **Pre-registration:** carry the cohort-1 claims (balance tracks recovery; larger lick responses; partial remapping;
  correction-based compensation if 2.1 holds) as pre-registered hypotheses for cohort 2, never pooled.

## References (verified from a page the helpers saw unless marked)
Bernhardt et al. 2017 Int J Stroke 12:444 (SRRR). Levin, Kleim & Wolf 2009 Neurorehabil Neural Repair 23:313. Murphy &
Corbett 2009 Nat Rev Neurosci 10:861. Krakauer et al. 2012 NNR 26:923. Zeiler & Krakauer 2013 Curr Opin Neurol 26:609.
Hope et al. 2019 Brain 142:15. Brown et al. 2007 J Neurosci 27:4101; 2009 J Neurosci 29:1719. Harrison et al. 2013
Stroke 44:2300. Lim et al. 2014 J Neurosci 34:16455. Latifi et al. 2020 Cereb Cortex 30:6363. Spalletti et al. 2017
eLife 6:e28662. Allegra Mascaro et al. 2019 Cell Rep 28:3474. Balbi et al. 2019 JCBFM 39:1486. Bauer et al. 2014
NeuroImage 99:388. Cramer et al. 2019 NeuroImage 199:570. Bice et al. 2022 eLife 11:e68852. Cecchini et al. 2021 PLoS
Comput Biol 17:e1008963. Castro-Alamancos & Borrell 1995 Neuroscience 68:793. Zeiler et al. 2016 NNR 30:794. Makin &
Krakauer 2023 eLife 12:e84716. Di Pino et al. 2014 Nat Rev Neurol 10:597 [partial]. Murase et al. 2004 Ann Neurol 55:400.
Grefkes et al. 2008 Ann Neurol 63:236. Ward et al. 2003 Brain 126:2476. Carrera & Tononi 2014 Brain 137:2408. Arvidsson et
al. 2002 Nat Med 8:963. Ruscher et al. 2012 Stroke 43:507. Ng et al. 2015 Stroke 46:2951. Clarkson et al. 2010 Nature
468:305. Joy & Carmichael 2021 Nat Rev Neurosci 22:38. Ramanathan et al. 2018 Nat Med 24:1257. Khanna et al. 2021 Cell
184:912. Guo et al. 2021 Cell Rep 36:109370. Lemke et al. 2019 Nat Neurosci 22:1122. Heimbuch et al. 2026 bioRxiv
10.64898/2026.02.10.705199. Driscoll et al. 2017 Cell 170:986. bioRxiv 10.1101/2025.10.31.685962 (internal-capsule stroke,
no cortical reorganisation) and 10.1101/2025.11.24.690099 (contralesional activity = compensation) [authors unverified].
Lee, Wang & Sabatini 2020 Nat Neurosci 23:1388. Lee & Sabatini 2021 Nature 599:645. Chen et al. 2021 Cell Rep 37:109847.
Rossi et al. 2016 Nat Neurosci 19:742. Thomas et al. 2023 Nat Commun 14. Ito et al. 2025 Nature 637:1143. Stanek et al.
2014 eLife 3:e02511. Dempsey et al. 2021 Nat Commun 12:6307. Guo et al. 2014 Neuron 81:179. Li et al. 2015 Nature 519:51.
Li et al. 2016 Nature 532:459. Economo et al. 2018 Nature 563:79. Guo et al. 2017 Nature 545:181. Inagaki et al. 2022 Cell
185:1065. Bollu et al. 2021 Nature 594:82. Xu et al. 2022 Nature 603:464. Mayrhofer et al. 2019 Neuron 103. Esmaeili et al.
2021 Neuron. Musall et al. 2019 Nat Neurosci 22:1677. Stringer et al. 2019 Science 364:eaav7893. Steinmetz et al. 2019
Nature 576:266. Kawai et al. 2015 Neuron 86:800. Dhawale et al. 2021 Nat Neurosci 24:1256. Yttri & Dudman 2016 Nature
533:402. Skitek et al. 1999 Behav Neurosci 113:567. Gallego et al. 2018 Nat Commun 9:4233; 2020 Nat Neurosci 23:260.
Degenhart et al. 2020 Nat Biomed Eng 4:672. Safaie et al. 2023 Nature 623:765. Sadtler et al. 2014 Nature 512:423.
Kaufman et al. 2014 Nat Neurosci 17:440. Perich et al. 2018 Neuron 100:964. Elsayed et al. 2016 Nat Commun 7:13239. Hasnain
et al. 2025 Nat Neurosci (10.1038/s41593-024-01859-1). Williams et al. 2018 Neuron 98:1099. Hamaker et al. 2015 Psychol
Methods 20:102. Harris 2020 bioRxiv 10.1101/2020.11.29.402719. Yu et al. 2022 Neuron 110:21.

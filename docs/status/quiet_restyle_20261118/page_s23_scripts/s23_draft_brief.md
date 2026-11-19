# Brief: draft section-23 caption rewrites (shared by all drafters)

WRITE ONLY: your part file `scratchpad/s23_part_<X>.py`, images under `scratchpad/s23img/`, and the harness output `scratchpad/s23_check_s23_part_<X>.html`.
- Edit nothing else: no repo, no M: file, and not s23_edit.py or s23_page.html.
- Publish nothing and send no messages.
- Kill nothing; start no long jobs.
- Python: `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe` (bare python fails).

SCRATCH = C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad
R = /m/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118

## Task
The PI approved moving these figures onto the new sync-end run, conclusion changes included. Inputs:
- Current page captions: SCRATCH/s23cmp/f-<fid>.txt
- Old-vs-new audit: SCRATCH/s23cmp/out_A.md, out_B.md, out_C.md, out_D.md
- New logs: R/logs/
- Bootstrap star tables: R/<family>/<name>_boot_stars.csv
  - Columns: panel, epoch, point, lo, hi, k_same, n_animals, per_animal, boot_star, lmm_q.
  - These are the AUTHORITATIVE significance calls: `*` = boot_star; `†` = lmm_q < .05 without a boot star.
  - Some pngs were re-plotted with these stars and have a boot_stars csv beside them. Others have none (maps, attempt_effort); see the per-figure notes.

For each figure, write in your part file:
```
set_img("<old>.png", "<old>_syncend.png")
set_title(fid, old_title_text, new_title_text, short=...)   # only if the title's claim is no longer true
set_caption(fid, "<full new caption html>")
```
- Copy the new png to SCRATCH/s23img/<old>_syncend.png.
- If one old image now maps to several new pngs, stack them vertically into one composite png with PIL (white background, max width ~2400 px).
- Find the exact old title text in SCRATCH/s23_page.html. It is the text inside `class="ftitle"` after the tag span. set_title needs it to occur at least twice in the figure (title + alt).

The part file is exec'd after the helpers and constants in SCRATCH/s23_check.py, so these names are available: rep, set_caption, set_img, set_title, data_block(extra), STARS, RUN, QF, LK, TH, SESS.

Test with: `cd SCRATCH && <python> s23_check.py s23_part_<X>.py`. It must print "ok".

## Caption rules (PI's standing rules; follow exactly)
- REPLACE the caption. Don't annotate it: no "previously", "was", "now", "corrected" or old-vs-new numbers in the text. Write it as if fresh.
- Start with `<b class="h">Fig 23.N &middot; NEURAL ... &middot; short descriptor.</b>`, keeping the existing tag words.
- Give per-animal values first (PS92, PS93, PS94, PS95), then the pooled change with its 95% CI [lo, hi] and k/4.
  - Every readout is the change vs the animal's OWN pre.
  - Mark `*` only where boot_star is set; mark `&dagger;` only for LMM-only significance.
- Include provenance:
  - window, lick source, quiet-frame definition, threshold and baseline: use `data_block()`;
  - n animals / sessions, and any sessions dropped and why;
  - the script path (`widefield_pipeline <code>scripts/cohort2_qc/<script>.py</code>`) and the stats csv names relative to the run folder.
- Describe the png as it actually looks:
  - These are the scripts' own figure layouts, not the restyled open-marker style.
  - Use STARS only if the png really shows open session markers etc. Otherwise write one sentence describing the markers, with the star rule: * = nested animal→session bootstrap 95% CI of the change excludes 0, † = mixed model q < .05 backup, stars in the top band.
- Drop sentences whose numbers no longer exist; keep what still holds.
- Remove any mention of the old lick rule ("cam1-only labels far from the mouth ... counts as licks") and of `v2_quiet_lp/` or `v2_quiet/` paths.
  - The new pngs must cite only the new run folder.
- American spelling. Use HTML entities like the existing captions (&minus; &mdash; &times; &rarr;).
- End with "PROVISIONAL (n = 4)." as the other captions do.
- If a number is not in the logs/csvs, leave it out. Don't compute new statistics yourself, except simple arithmetic on printed numbers, and say so in your report.

## Report back
For each figure:
- the new title (if changed);
- a 2–4 line summary of the conclusion text;
- anything you were unsure of;
- the harness result.

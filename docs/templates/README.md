# Results pages (Priya 10-24)

Every result family gets a published Artifact page for review. `results_page_example.html` is the worked example (the
goal-question page, https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d). Conventions:

- **Left-panel table of contents**, clickable and hidable (remembers open / closed per viewer; overlays on phones).
  Built by the small script at the bottom of the example from `<section id=... data-toc="..." data-kind="n|l">` and
  `<figure id=...>` elements -- copy the CSS block, the `#tocbtn` / `#toc` markup and the script.
- **Figures numbered SECTION.POSITION** (Priya 10-26, all pages; was Fig 1, 2b, 4c ...): sections headed
  `<h2>N. ...`, figures Fig N.1, N.2 ... in page order, ids `fig<N>-<k>`, caption label
  `<figcaption><b class="h">Fig N.k &middot; short label.</b> ...`. Every in-page link and "Fig X" cross reference uses
  N.k (remap after any move; check for broken links). Reference implementation: the goal page's
  `renumber_sections.py`; the FR page resolves `<a href="#src:fig/x.png">` placeholders at build so figures can move.
- **A TITLE above every image** (Priya 10-26): `<div class="ftitle"><span class="fnum">Fig N.k</span>title</div>` naming
  the model / data, window and the one-line finding ("Frozen pre-stroke encoder pixel R² maps, no FaceRhythm: acute
  full-R² loss carried by movement unique"); the contents lists number + title.
- **Captions carry the full method** (Priya 10-26): model / decoder (frozen or refit) or data source (pose model, camera),
  window / bins and baseline, variables included / excluded, trial / lick gate, sessions / animals, statistic, correction
  and n, and the script (`<b>Script:</b> <code>...</code>`).
- **Every caption ends with the figure's FILE NAME** (Priya 11-09, all pages: "we should make including file names the
  default for all pages"): `<span class="ffile"><b>File:</b> <code><share folder>/<image></code></span>` as the last item
  of the `<figcaption>`, from the figure's current `<img src>`. Run `scripts/page_file_names.py <index.html> --prefix
  <share folder>` after every build or figure swap (idempotent; first used on the state-space page v35).
- **Tag each section** neural (`data-kind="n"`) or lick / behavior (`data-kind="l"`).
- Each figure's caption says how to read it and gives the headline numbers; one `.take` box per section with the
  result; within a section the pooled / cohort (within-animal-normalized, with statistics) figures come BEFORE the
  per-animal ones (Priya 10-25 evening; reverses the earlier per-animal-first rule), headline figure first.
  Change maps: diverging RdBu_r (red = more), one color scale for all panels of a figure, significant points large and
  outlined with the rest faded.
- **Significance stars go in a band at the TOP of each panel** (Priya 10-30, all figures): visible at a glance, never on
  data points, error bars or the horizontal pre / 0 line. Use `wfield_local.sig_marks.star_top` + `reserve_headroom`
  (or `orofacial_cohort._mark_rows` / `_tidy`), not `ax.text(x, ci_hi + pad, "*")`.
- Section tags point to the FINDINGS section the result lives in.
- **"Where things stand" table at the top cites its evidence** (Priya 10-25): a "figures" column right after the result,
  each cell linking the supporting figures (`<a href="#fig7b">7b</a>`), or the section when only a table supports it.
- **Windows are always ordered PRE-CUE (ENL), CUE, LICK** (Priya 10-25, "across figures and pages"): figures within a
  section, rows / columns / panels inside a figure, table columns, sidebar entries.
- **Like figures sit together**, nested WINDOW > REFERENCE (rest, precue ...) > GATE (DAQ contact next to LP lick), and
  every map's title names all three ("LICK window · REST reference · LP-lick gate · pooled") so no title relies on the
  figure above it (Priya 10-25).
- Figures appear in number / letter order down the page; renumber rather than insert out of sequence. The contents
  groups each figure number's letters behind a toggle (the encoding page's script).

## Figure color / marker defaults (Priya 10-26, 10-30)
- Epoch colors everywhere = `wfield_local.orofacial_cohort.EPOCH_LINE`: pre `k`, acute `#d01c8b`, subacute `#b8860b`,
  chronic `#01665e`; a no-stroke / June reference = gray.
- Show SESSIONS, not per-animal lines: open markers, shape = animal (`orofacial_cohort.ANIMAL_MARKER`: PS92 o, PS93 s,
  PS94 ^, PS95 D), color = epoch; pooled level = filled marker / thick line + 95 % CI in the epoch color.
- Distributions (ECDFs, histograms): thin per-session curves in epoch colors + thick pooled curve per epoch.
- Panels comparing several series (hit vs miss, hemispheres): keep series colors, epoch on the x axis, say so in the
  caption.

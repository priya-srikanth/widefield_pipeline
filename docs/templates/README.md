# Results pages (Priya 10-24)

Every result family gets a published Artifact page for review. `results_page_example.html` is the worked example (the
goal-question page, https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d). Conventions:

- **Left-panel table of contents**, clickable and hidable (remembers open / closed per viewer; overlays on phones).
  Built by the small script at the bottom of the example from `<section id=... data-toc="..." data-kind="n|l">` and
  `<figure id=...>` elements -- copy the CSS block, the `#tocbtn` / `#toc` markup and the script.
- **Every figure numbered** (Fig 1, 2b, 4c ...) in a `<figcaption><b class="h">Fig N &middot; title.</b> ...` -- the
  contents lists the bold part.
- **Tag each section** neural (`data-kind="n"`) or lick / behaviour (`data-kind="l"`).
- Each figure's caption says how to read it and gives the headline numbers; one `.take` box per section with the
  result; per-animal first, then within-animal-normalised pooled results with statistics.
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

## Figure colour / marker defaults (Priya 10-26, 10-30)
- Epoch colours everywhere = `wfield_local.orofacial_cohort.EPOCH_LINE`: pre `k`, acute `#d01c8b`, subacute `#b8860b`,
  chronic `#01665e`; a no-stroke / June reference = grey.
- Show SESSIONS, not per-animal lines: open markers, shape = animal (`orofacial_cohort.ANIMAL_MARKER`: PS92 o, PS93 s,
  PS94 ^, PS95 D), colour = epoch; pooled level = filled marker / thick line + 95 % CI in the epoch colour.
- Distributions (ECDFs, histograms): thin per-session curves in epoch colours + thick pooled curve per epoch.
- Panels comparing several series (hit vs miss, hemispheres): keep series colours, epoch on the x axis, say so in the
  caption.

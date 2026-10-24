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

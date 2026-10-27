from pathlib import Path

p = Path(r"C:\Users\SabatiniLab\AppData\Local\Temp\claude\C--Users-SabatiniLab\c67bec01-3925-49ff-9cdc-8ff295df4bfa\scratchpad\imaging_page\imaging_results.html")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep(" A June-vs-August pre map-drift version is in progress (e3).</p>",
    " Fig 7k compares the stroke changes with map drift between June and August pre-stroke sessions (no stroke).</p>")
anchor = 'the far-contra loss seen after the cue is not present in the lick-free pre-cue window; chronic far contra falls below pre.</figcaption></figure>'
rep(anchor, anchor + "\n" +
    '<figure id="fig7k"><img src="p_drift_rest_cue.png" alt="Map drift June to August pre beside stroke changes, rest reference, cue, contact" loading="lazy"><figcaption><b class="h">Fig 7k &middot; map drift without a stroke (June &rarr; August pre) beside the stroke changes &mdash; REST reference, cue window, contact gate.</b> '
    'Columns: June pre | August pre | Aug &minus; June (no stroke; scaled by June maps) | acute &minus; Aug pre | chronic &minus; Aug pre (scaled by August maps). Same maps and test as Fig 7b; pooled within animal; contours = nested animals &rarr; sessions bootstrap; 4 animals, 12 June / 32 August sessions. Ratio = amplitude vs the reference epoch. '
    '<b>Far contra:</b> drift is small (Aug / June .95, a handful of significant bins) against the acute loss (.49, large bilateral contours); chronic 1.02 with lateral frontal / motor increases &mdash; drift does not explain the far-contra acute loss. '
    '<b>Far middle is the exception:</b> drift (.77, with contours) is as large as the acute change (.71), so the far-middle acute loss cannot be separated from drift. '
    '<b>Other positions:</b> drift .76&ndash;.94, mostly small decreases; their stroke changes are mostly increases (near ipsi acute 1.51, chronic 1.43) &mdash; opposite in sign to the drift. Chronically a bilateral anterolateral (ALM / MOs-like) increase appears at most positions. '
    'Caveat: June still includes learning, so the drift column is an upper bound on drift alone; June has ~3 sessions per animal vs ~8 in August. Script <code>scripts/map_drift_prepre.py</code> (ccf631a); the LP, pre-cue-reference and ENL arms are rendering.</figcaption></figure>')
p.write_text(s, encoding="utf-8")
print("ok")

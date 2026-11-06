import re
from pathlib import Path

p = Path(r"C:\Users\SabatiniLab\AppData\Local\Temp\claude\C--Users-SabatiniLab\c67bec01-3925-49ff-9cdc-8ff295df4bfa\scratchpad\imaging_page\imaging_results.html")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


POOL = ("Pooled over 4 animals; difference columns within animal (each animal minus its own pre); "
        "contours = nested animals &rarr; sessions bootstrap, max-statistic corrected, eroded mask.")
PA = ("Far-contra position for each animal; colour scale per animal; no significance drawn "
      "(a per-animal panel has no between-animal spread to test).")


def fig(fid, src, title, body):
    return ('<figure id="%s"><img src="%s" alt="%s" loading="lazy"><figcaption><b class="h">Fig %s &middot; %s.</b> %s</figcaption></figure>'
            % (fid, src, title.replace("&middot;", "-").replace("&rarr;", "to"), fid[3:].replace("_", "."), title, body))


C, L = "contact gate", "LP-lick gate"
D = "June &rarr; Aug drift"
figs = {}
figs["cue_rest"] = [
    fig("fig7_2a", "map_REST_cue_contact.png", "CUE window &middot; REST reference &middot; %s &middot; pooled" % C,
        POOL + " Acute amplitude / own pre: far contra .48, far middle .66, near ipsi 1.43. LP twin: Fig 7.2c."),
    fig("fig7_2b", "map_REST_cue_contact_by_animal.png", "CUE window &middot; REST reference &middot; %s &middot; per animal" % C, PA),
    fig("fig7_2c", "map_REST_cue_LP.png", "CUE window &middot; REST reference &middot; %s &middot; pooled" % L, POOL + " Contact twin: Fig 7.2a."),
    fig("fig7_2d", "drift_REST_cue_contact.png", "CUE window &middot; REST reference &middot; %s &middot; %s" % (C, D),
        "Columns: June pre | August pre | Aug &minus; June (no stroke; scaled by June maps) | acute &minus; Aug pre | chronic &minus; Aug pre (scaled by August maps). Same maps and test as Fig 7.2a; 4 animals, 12 June / 32 August sessions; ratio = amplitude vs the reference epoch. "
        "<b>Far contra:</b> drift is small (Aug / June .95, 7 of 2,022 bins significant) against the acute loss (.49, 1,178 bins); chronic 1.02 &mdash; drift does not explain the far-contra acute loss. "
        "<b>Far middle is the exception:</b> drift (.77, 244 bins) is as large as the acute change (.71, 151 bins), so the far-middle acute loss cannot be separated from drift in the cue window (not in the lick window: Fig 7.3e). Far ipsi drifts as much (.76, 294 bins) but shows no acute change (1.06, 6 bins). "
        "<b>Other positions:</b> drift .76&ndash;.94, mostly small decreases; their stroke changes are mostly increases (near ipsi acute 1.51, chronic 1.43), opposite in sign. Chronically a bilateral anterolateral (ALM / MOs-like) increase appears at most positions. "
        "Caveat: June still includes learning, so this is an upper bound on drift alone; June ~3 sessions per animal vs ~8 in August. <code>scripts/map_drift_prepre.py</code> (ccf631a). Contact arm complete (10-25)."),
]
figs["cue_pre"] = [
    fig("fig7_2e", "map_PRECUE_cue_contact.png", "CUE window &middot; PRECUE reference &middot; %s &middot; pooled" % C,
        POOL + " Acute amplitude / own pre: far contra .18, far middle .52, far ipsi 1.00, near positions 1.55&ndash;1.69. The cue-evoked increment falls more than the REST-referenced map (Fig 7.2a, .48); chronic far contra is above pre. LP twin: Fig 7.2f."),
    fig("fig7_2f", "map_PRECUE_cue_LP.png", "CUE window &middot; PRECUE reference &middot; %s &middot; pooled" % L, POOL + " Contact twin: Fig 7.2e."),
    fig("fig7_2g", "drift_PRECUE_cue_contact.png", "CUE window &middot; PRECUE reference &middot; %s &middot; %s" % (C, D),
        "Same layout as Fig 7.2d. Ratio, significant bins of 2,022 in brackets: drift | acute | chronic. "
        "<b>Far contra:</b> drift 1.10 (33) | acute .18 (1,883) | chronic 1.14 (230). <b>Far middle:</b> .83 (30) | .55 (800) | 1.51 (295). "
        "<b>Far ipsi:</b> .85 (2) | 1.07 (0). <b>Near:</b> drift .84&ndash;1.02 (2&ndash;48) | acute 1.61&ndash;1.79 (0&ndash;36). "
        "Under the PRECUE reference both the far-contra and the far-middle acute losses are far beyond drift."),
]
figs["lick_rest"] = [
    fig("fig7_3a", "map_REST_lick_contact.png", "LICK window &middot; REST reference &middot; %s &middot; pooled" % C,
        POOL + " <b>Cannot show the headline far-contra loss:</b> the far-contra acute panel is empty because too few acute far-contra licks reach the spout to pass the 20-trial floor; far middle acute has 3 animals, 11 sessions, n = 500, no contours. Near positions look almost the same as the LP version because most of their trials are shared. LP twin (the headline lick figure): Fig 7.3c."),
    fig("fig7_3b", "drift_REST_lick_contact.png", "LICK window &middot; REST reference &middot; %s &middot; %s" % (C, D),
        "Contact twin of Fig 7.3e; columns as Fig 7.2d. Significant bins of 2,022 in brackets (R = rim-suppressed): far contra drift 1.11 (41), chronic 1.01 (298); far middle .93 (6, R) vs acute 1.29 (8, R); far ipsi 1.00 (77, R) vs 1.42 (177); near drift .86&ndash;.98 (9&ndash;33) vs acute 1.32&ndash;1.58 (181&ndash;661). "
        "<b>Far contra:</b> drift 1.11 (a few small frontal contours); acute has no data (same 20-trial floor as Fig 7.3a); chronic 1.01 &mdash; for the far-contra lick result use the LP-lick drift, Fig 7.3e. "
        "<b>Far middle:</b> drift .93 vs acute 1.29 (3 animals), no loss. <b>Far ipsi:</b> drift 1.00 vs acute 1.42. "
        "<b>Near positions:</b> drift .86 / .88 / .98 vs acute 1.58 / 1.32 / 1.36 &mdash; drift small, stroke changes go up."),
    fig("fig7_3c", "map_REST_lick_LP.png", "LICK window &middot; REST reference &middot; %s &middot; pooled" % L,
        POOL + " <b>Headline lick figure.</b> Far contra acute .67 (15 sessions, n = 788 trials), a large significant loss mostly ipsilesional; back to 1.02 subacute, 1.03 chronic. The LP gate keeps the acute far-contra attempts that never reach the spout, which the contact gate (Fig 7.3a) loses. <span class=\"st prov\">provisional</span>"),
    fig("fig7_3d", "map_REST_lick_LP_by_animal.png", "LICK window &middot; REST reference &middot; %s &middot; per animal" % L,
        PA + " Far contra acute per animal &minus;55 / &minus;48 / &minus;38 / +2 % (PS95 one session)."),
    fig("fig7_3e", "drift_REST_lick_LP.png", "LICK window &middot; REST reference &middot; %s &middot; %s" % (L, D),
        "Columns as Fig 7.2d: June pre | August pre | Aug &minus; June (no stroke) | acute &minus; Aug pre | chronic &minus; Aug pre. Ratio = amplitude vs the reference epoch; bins = significant of 2,022. "
        "<b>Far contra:</b> drift 1.04 (15 bins) against acute .67 (500 bins); chronic 1.03 &mdash; the acute far-contra loss is far beyond drift. "
        "<b>Far middle:</b> drift .90 (12 bins, suppressed as a rim artefact), acute 1.07 (34 bins) &mdash; no far-middle acute loss in the lick window, so the cue-window caveat (Fig 7.2d) does not apply here. "
        "<b>Far ipsi:</b> drift .87 (116 bins), acute 1.33 (101). <b>Near positions:</b> drift .86&ndash;.98 (&le; 26 bins), acute increases 1.32&ndash;1.58 (185&ndash;667 bins), opposite in sign to drift. "
        "PS93_0813 is missing (no behaviour video, so no LP lick table). <code>map_drift_prepre/LP_licks/drift_vs_stroke_panels.csv</code>."),
]
figs["lick_pre"] = [
    fig("fig7_3f", "map_PRECUE_lick_contact.png", "LICK window &middot; PRECUE reference &middot; %s &middot; pooled" % C, POOL + " LP twin: Fig 7.3h."),
    fig("fig7_3g", "drift_PRECUE_lick_contact.png", "LICK window &middot; PRECUE reference &middot; %s &middot; %s" % (C, D),
        "Contact twin of Fig 7.3f; columns as Fig 7.2d. Significant bins of 2,022 in brackets: far contra drift 1.32 (86), chronic 1.09 (172); far middle 1.02 (0) vs acute 1.34 (0); far ipsi 1.08 (90) vs 1.58 (189); near drift .86&ndash;1.05 (0&ndash;38) vs acute 1.78&ndash;1.92 (6&ndash;53). "
        "<b>Far contra:</b> drift 1.32 <i>up</i>, with medial-frontal contours &mdash; the cue-evoked lick increment at far contra grew from June to August, consistent with learning, so under the PRECUE reference drift runs opposite to the stroke loss; acute has no data (contact floor); chronic 1.09. "
        "<b>Far middle:</b> drift 1.02 vs acute 1.34 (3 animals). <b>Far ipsi:</b> drift 1.08 (medial-frontal contour) vs acute 1.58. "
        "<b>Near positions:</b> drift .86 / .94 / 1.05 vs acute 1.92 / 1.78 / 1.88."),
    fig("fig7_3h", "map_PRECUE_lick_LP.png", "LICK window &middot; PRECUE reference &middot; %s &middot; pooled" % L, POOL + " Contact twin: Fig 7.3f."),
]
figs["pc_rest"] = [
    fig("fig7_1a", "map_REST_precue_contact.png", "PRE-CUE 2 s window &middot; REST reference &middot; %s &middot; pooled" % C,
        POOL + " The anticipatory position signal before the cue. LP twin: Fig 7.1c."),
    fig("fig7_1b", "drift_REST_precue_contact.png", "PRE-CUE 2 s window &middot; REST reference &middot; %s &middot; %s" % (C, D),
        "Contact twin of Fig 7.1a; columns as Fig 7.2d. Ratio, significant bins of 2,022 in brackets (R = rim-suppressed). "
        "<b>Far contra:</b> drift .67 (67) | acute 1.85 <i>up</i> (339) | chronic .65 (0) &mdash; the far-contra pre-cue map falls by a third with time alone, so the chronic .65 is drift-sized, not a stroke effect; the acute increase is beyond drift. "
        "<b>Far middle:</b> drift .89 (37, R) | acute .96 (41) | chronic .53 (572). <b>Near middle:</b> drift .77 (30) | acute .55 (6) | chronic .69 (316). "
        "Other positions: drift .66&ndash;.74, few bins (most suppressed or 0)."),
    fig("fig7_1c", "map_REST_precue_LP.png", "PRE-CUE 2 s window &middot; REST reference &middot; %s &middot; pooled" % L, POOL + " Contact twin: Fig 7.1a."),
]
figs["enl"] = [
    fig("fig7_1d", "map_REST_enl1s_contact.png", "ENL 1 s window &middot; REST reference &middot; %s &middot; pooled" % C,
        POOL + " ENL = the 1 s lick-free window before the cue, on trials with no DAQ lick in it; because the trials are lick-free the gate barely matters, so there is no LP twin. Acute amplitude / own pre: far contra 1.67, far ipsi 1.10, far middle 1.00, near ipsi 1.72, near middle .42, near contra .55 &mdash; the far-contra loss seen after the cue is not present here; chronic far contra falls below pre. June &rarr; Aug drift for ENL will follow when it renders."),
]


def link(fid):
    return '<a href="#%s">%s</a>' % (fid, fid[3:].replace("_", "."))


idx = """<div class="tbl"><table>
<tr><th>window &middot; reference</th><th>contact gate</th><th>LP-lick gate</th><th>June &rarr; Aug drift</th></tr>
<tr><td>PRE-CUE 2 s &middot; REST</td><td>%s</td><td>%s</td><td>%s (contact); LP pending</td></tr>
<tr><td>ENL 1 s &middot; REST</td><td>%s</td><td>no twin (lick-free trials)</td><td>pending</td></tr>
<tr><td>CUE &middot; REST</td><td>%s, %s per animal</td><td>%s</td><td>%s (contact); LP pending</td></tr>
<tr><td>CUE &middot; PRECUE</td><td>%s</td><td>%s</td><td>%s (contact); LP pending</td></tr>
<tr><td>LICK &middot; REST</td><td>%s</td><td>%s (headline), %s per animal</td><td>%s (contact), %s (LP-lick)</td></tr>
<tr><td>LICK &middot; PRECUE</td><td>%s</td><td>%s</td><td>%s (contact); LP pending</td></tr>
</table></div>""" % tuple(link(x) for x in [
    "fig7_1a", "fig7_1c", "fig7_1b", "fig7_1d",
    "fig7_2a", "fig7_2b", "fig7_2c", "fig7_2d", "fig7_2e", "fig7_2f", "fig7_2g",
    "fig7_3a", "fig7_3c", "fig7_3d", "fig7_3b", "fig7_3e", "fig7_3f", "fig7_3h", "fig7_3g"])

m = re.search(r'<section id="maps".*?</section>', s, re.S)
old = m.group(0)
i0 = old.index('<div class="tbl">')
i1 = old.index("</div>", old.index('<div class="take">')) + len("</div>")
keep = old[i0:i1]
keep = keep.replace("<th>far contra amplitude / pre</th>", "<th>far contra amplitude / pre (RESTW reference)</th>")
keep = keep.replace("Caveat: the pre-stroke rest SD",
                    "The cue numbers in this table and takeaway come from the position-weighted rest (RESTW) maps; the time-local REST maps used below give .48 / .66 / 1.43 (Fig 7.2a), the same conclusion. Caveat: the pre-stroke rest SD")
new = ('<section id="maps" data-toc="7. Maps: far-contra amplitude" data-kind="n">\n'
       '<span class="tag">Anatomy &middot; position maps by window, reference and gate</span>\n'
       '<h2>7. The far-contra cortical response roughly halves acutely and returns <span class="kind n">neural</span><span class="st stands">stands</span></h2>\n'
       '<p class="input">Every map is titled <b>window &middot; reference &middot; gate</b>. Windows, always in this order: PRE-CUE 2 s, ENL 1 s (lick-free pre-cue), CUE (0&ndash;2 s after the cue), LICK. References: REST = each session&rsquo;s time-local rest baseline; PRECUE = each trial&rsquo;s own pre-cue window, so it shows the cue-evoked <i>increment</i> and removes the anticipatory position signal (no PRECUE version of the pre-cue windows, by construction). Gates: contact = DAQ spout-contact trials; LP-lick = Lightning Pose lick-attempt trials. Figures nest window &gt; reference &gt; gate, with contact and LP side by side (renumbered 10-25).</p>\n'
       + keep + "\n" + idx + "\n"
       '<h3 class="h7">7.1 PRE-CUE windows</h3>\n<h4 class="h7">7.1.1 PRE-CUE 2 s window, REST reference</h4>\n' + "\n".join(figs["pc_rest"]) + "\n"
       '<h4 class="h7">7.1.2 ENL 1 s window, REST reference</h4>\n' + "\n".join(figs["enl"]) + "\n"
       '<h3 class="h7">7.2 CUE window</h3>\n<h4 class="h7">7.2.1 REST reference</h4>\n' + "\n".join(figs["cue_rest"]) + "\n"
       '<h4 class="h7">7.2.2 PRECUE reference</h4>\n' + "\n".join(figs["cue_pre"]) + "\n"
       '<h3 class="h7">7.3 LICK window</h3>\n<h4 class="h7">7.3.1 REST reference</h4>\n' + "\n".join(figs["lick_rest"]) + "\n"
       '<h4 class="h7">7.3.2 PRECUE reference</h4>\n' + "\n".join(figs["lick_pre"]) + "\n"
       "</section>")
s = s.replace(old, new)

rep("(Fig 7k), so treat it as not separable from drift.", "(Fig 7.2d), so treat it as not separable from drift.")
rep("h2 { font:600 1.15rem/1.3",
    "h3.h7 { font:600 1.02rem/1.3 \"Source Serif 4\", Georgia, serif; margin:26px 0 6px; padding-top:8px; border-top:1px solid var(--rule) }\n"
    "h4.h7 { font:600 .8rem/1.3 \"Source Sans 3\", sans-serif; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:16px 0 6px }\n"
    "h2 { font:600 1.15rem/1.3")
# TOC: group by sub-number (fig7_2a.. -> one group per window)
rep("var m = /^fig(\\d+)/.exec(f.id); var num = m ? m[1] : f.id;",
    "var m = /^fig(\\d+(?:_\\d+)?)/.exec(f.id); var num = m ? m[1] : f.id;")
p.write_text(s, encoding="utf-8")
left = re.findall(r"Fig 7[a-n]\b", s)
print("ok; stale Fig 7x refs:", left)

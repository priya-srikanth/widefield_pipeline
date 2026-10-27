"""Section 7 of the imaging page, generated from symbolic figure keys.

Figure numbers are assigned in page order and every {key} in a caption is resolved to its number, so inserting a
figure never needs a manual renumber. Run on the v12 backup, then p_img_winorder.py and p_img_s1face.py.
"""
import re
from pathlib import Path

p = Path(__file__).parent / "imaging_page" / "imaging_results.html"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


POOL = ("Pooled over 4 animals; difference columns within animal (each animal minus its own pre); "
        "contours = nested animals &rarr; sessions bootstrap, max-statistic corrected, eroded mask.")
PA = ("Far-contra position for each animal; colour scale per animal; no significance drawn "
      "(a per-animal panel has no between-animal spread to test).")
DRIFTCOLS = ("Columns: June pre | August pre | Aug &minus; June (no stroke; scaled by June maps) | acute &minus; Aug pre | "
             "chronic &minus; Aug pre (scaled by August maps); same maps and test as the pooled map; 4 animals, 12 June / 32 August sessions. "
             "Ratio = amplitude vs the reference epoch; significant bins of 2,022 in brackets (R = rim-suppressed). ")
GATE = {"C": "contact gate", "L": "LP-lick gate"}
WIN = {"pc": "PRE-CUE 2 s window", "enl": "ENL 1 s window", "cue": "CUE window", "lick": "LICK window"}
REF = {"R": "REST reference", "P": "PRECUE reference"}
KIND = {"pool": "pooled", "pa": "per animal", "drift": "June &rarr; Aug drift"}

# (key, file, window, reference, gate, kind, caption)
F = {}
LAYOUT = []   # (heading html, [keys]) in page order


def f(key, src, w, r, g, k, body):
    F[key] = dict(src=src, w=w, r=r, g=g, k=k, body=body)
    return key


pc = [
    f("pcRC", "map_REST_precue_contact.png", "pc", "R", "C", "pool", POOL + " The anticipatory position signal before the cue. LP twin: Fig {pcRL}."),
    f("pcRCd", "drift_REST_precue_contact.png", "pc", "R", "C", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift .67 (67) | acute 1.85 <i>up</i> (339) | chronic .65 (0) &mdash; the far-contra pre-cue map falls by a third with time alone, so the chronic .65 is drift-sized, not a stroke effect; the acute increase is beyond drift. "
      "<b>Far middle:</b> drift .89 (37, R) | acute .96 (41) | chronic .53 (572) &mdash; the chronic loss is beyond drift. <b>Near middle:</b> drift .77 (30) | acute .55 (6) | chronic .69 (316). "
      "Other positions: drift .66&ndash;.74, few bins (most suppressed or 0)."),
    f("pcRL", "map_REST_precue_LP.png", "pc", "R", "L", "pool", POOL + " Contact twin: Fig {pcRC}."),
    f("pcRLd", "drift_REST_precue_LP.png", "pc", "R", "L", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift .65 (37) | acute 1.84 (125) | chronic .81 (0) &mdash; same picture as contact (Fig {pcRCd}). "
      "<b>Far middle:</b> drift .97 (6, R) | acute .96 | chronic .30 (1,051) &mdash; chronic loss far beyond drift. "
      "<b>Near middle:</b> drift .73 | acute .55 (81) | chronic .68 (384)."),
]
enl = [
    f("enlRC", "map_REST_enl1s_contact.png", "enl", "R", "C", "pool", POOL +
      " ENL = the 1 s lick-free window before the cue, on trials with no DAQ lick in it; because the trials are lick-free the gate barely matters, so there is no LP twin. Acute amplitude / own pre: far contra 1.67, far ipsi 1.10, far middle 1.00, near middle .42, near contra .55 &mdash; the far-contra loss seen after the cue is not present here; chronic far contra falls below pre (but see the drift, Fig {enlRCd})."),
    f("enlRCd", "drift_REST_enl1s_contact.png", "enl", "R", "C", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift .63 (48) | acute 2.21 (186) | chronic .58 (0) &mdash; the same picture as the 2 s window (Fig {pcRCd}): the chronic fall is drift-sized. "
      "<b>Far middle:</b> drift .86 (25, R) | chronic .35 (954) &mdash; chronic loss beyond drift."),
]
cue_r = [
    f("cueRC", "map_REST_cue_contact.png", "cue", "R", "C", "pool", POOL + " Acute amplitude / own pre: far contra .48, far middle .66, near ipsi 1.43. LP twin: Fig {cueRL}."),
    f("cueRCa", "map_REST_cue_contact_by_animal.png", "cue", "R", "C", "pa", PA),
    f("cueRCd", "drift_REST_cue_contact.png", "cue", "R", "C", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift .95 (7) | acute .49 (1,178) | chronic 1.02 (319) &mdash; drift does not explain the far-contra acute loss. "
      "<b>Far middle is the exception:</b> drift .77 (244) is as large as the acute change .71 (151), so the far-middle acute loss cannot be separated from drift in the cue window (it can in the PRECUE reference, Fig {cuePCd}, and there is no far-middle acute loss in the lick window, Fig {lickRLd}). "
      "Far ipsi drifts as much, .76 (294), but shows no acute change, 1.06 (6, R). "
      "<b>Near positions:</b> drift .82&ndash;.94 (31&ndash;152), mostly small decreases; stroke changes are increases, 1.23&ndash;1.51 (81&ndash;469), opposite in sign. Chronically a bilateral anterolateral (ALM / MOs-like) increase appears at most positions. "
      "Caveat: June still includes learning, so this is an upper bound on drift alone; June ~3 sessions per animal vs ~8 in August. <code>scripts/map_drift_prepre.py</code>."),
    f("cueRL", "map_REST_cue_LP.png", "cue", "R", "L", "pool", POOL + " Contact twin: Fig {cueRC}."),
    f("cueRLd", "drift_REST_cue_LP.png", "cue", "R", "L", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift .94 (3) | acute .49 (1,181) | chronic 1.04 (315). <b>Far middle:</b> drift .78 (216) | acute .71 (136). "
      "Essentially identical to contact (Fig {cueRCd}): the gate does not matter in the cue window."),
]
cue_p = [
    f("cuePC", "map_PRECUE_cue_contact.png", "cue", "P", "C", "pool", POOL +
      " Acute amplitude / own pre: far contra .18, far middle .52, far ipsi 1.00, near positions 1.55&ndash;1.69. The cue-evoked increment falls more than the REST-referenced map (Fig {cueRC}, .48); chronic far contra is above pre. LP twin: Fig {cuePL}."),
    f("cuePCd", "drift_PRECUE_cue_contact.png", "cue", "P", "C", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift 1.10 (33) | acute .18 (1,883) | chronic 1.14 (230). <b>Far middle:</b> .83 (30) | .55 (800) | 1.51 (295). "
      "<b>Far ipsi:</b> .85 (2) | 1.07 (0). <b>Near:</b> drift .84&ndash;1.02 (2&ndash;48) | acute 1.61&ndash;1.79 (0&ndash;36). "
      "Under the PRECUE reference both the far-contra and the far-middle acute losses are far beyond drift."),
    f("cuePL", "map_PRECUE_cue_LP.png", "cue", "P", "L", "pool", POOL + " Contact twin: Fig {cuePC}."),
    f("cuePLd", "drift_PRECUE_cue_LP.png", "cue", "P", "L", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift 1.11 (34) | acute .18 (1,891). <b>Far middle:</b> drift .83 (24) | acute .54 (889). Same as contact (Fig {cuePCd})."),
]
lick_r = [
    f("lickRC", "map_REST_lick_contact.png", "lick", "R", "C", "pool", POOL +
      " <b>Cannot show the headline far-contra loss:</b> the far-contra acute panel is empty because too few acute far-contra licks reach the spout to pass the 20-trial floor; far middle acute has 3 animals, 11 sessions, n = 500, no contours. Near positions look almost the same as the LP version because most of their trials are shared. LP twin (the headline lick figure): Fig {lickRL}."),
    f("lickRCd", "drift_REST_lick_contact.png", "lick", "R", "C", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift 1.11 (41) | acute &mdash; (contact floor, as Fig {lickRC}) | chronic 1.01 (298); for the far-contra lick result use the LP-lick drift, Fig {lickRLd}. "
      "<b>Far middle:</b> drift .93 (6, R) | acute 1.29 (8, R), no loss. <b>Far ipsi:</b> 1.00 (77, R) | 1.42 (177). "
      "<b>Near positions:</b> drift .86&ndash;.98 (9&ndash;33) | acute 1.32&ndash;1.58 (181&ndash;661) &mdash; drift small, stroke changes go up."),
    f("lickRL", "map_REST_lick_LP.png", "lick", "R", "L", "pool", POOL +
      " <b>Headline lick figure.</b> Far contra acute .67 (15 sessions, n = 788 trials), a large significant loss mostly ipsilesional; back to 1.02 subacute, 1.03 chronic. The LP gate keeps the acute far-contra attempts that never reach the spout, which the contact gate (Fig {lickRC}) loses. <span class=\"st prov\">provisional</span>"),
    f("lickRLa", "map_REST_lick_LP_by_animal.png", "lick", "R", "L", "pa", PA + " Far contra acute per animal &minus;55 / &minus;48 / &minus;38 / +2 % (PS95 one session)."),
    f("lickRLd", "drift_REST_lick_LP.png", "lick", "R", "L", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift 1.04 (15) | acute .67 (500) | chronic 1.03 (314) &mdash; the acute far-contra loss is far beyond drift. "
      "<b>Far middle:</b> drift .90 (12, R) | acute 1.07 (34) &mdash; no far-middle acute loss in the lick window, so the cue-window caveat (Fig {cueRCd}) does not apply here. "
      "<b>Far ipsi:</b> .87 (116) | 1.33 (101). <b>Near positions:</b> drift .86&ndash;.98 (&le; 26) | acute 1.32&ndash;1.58 (185&ndash;667), opposite in sign to drift. "
      "PS93_0813 is missing (no behaviour video, so no LP lick table)."),
]
lick_p = [
    f("lickPC", "map_PRECUE_lick_contact.png", "lick", "P", "C", "pool", POOL + " LP twin: Fig {lickPL}."),
    f("lickPCd", "drift_PRECUE_lick_contact.png", "lick", "P", "C", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift 1.32 <i>up</i> (86), with medial-frontal contours &mdash; the cue-evoked lick increment at far contra grew from June to August, consistent with learning; acute has no data (contact floor); chronic 1.09 (172). "
      "<b>Far middle:</b> 1.02 (0) | 1.34 (0). <b>Far ipsi:</b> 1.08 (90) | 1.58 (189). <b>Near:</b> drift .86&ndash;1.05 (0&ndash;38) | acute 1.78&ndash;1.92 (6&ndash;53)."),
    f("lickPL", "map_PRECUE_lick_LP.png", "lick", "P", "L", "pool", POOL + " Contact twin: Fig {lickPC}."),
    f("lickPLd", "drift_PRECUE_lick_LP.png", "lick", "P", "L", "drift", DRIFTCOLS +
      "<b>Far contra:</b> drift 1.24 <i>up</i> (72) | acute .56 (436) | chronic 1.10 (186) &mdash; the only PRECUE lick panel with an acute far-contra estimate (contact is empty, Fig {lickPCd}); drift and stroke go in opposite directions, so the loss is beyond drift. "
      "<b>Near:</b> drift .86&ndash;1.05 | acute 1.73&ndash;1.88."),
]

TAKE71 = ('<div class="take"><b>Pre-cue windows (10-25 drift arms):</b> the chronic <b>far-middle</b> pre-cue map loss is beyond drift in every gate and window &mdash; chronic .53 (572 bins; 2 s contact), .30 (1,051; 2 s LP), .35 (954; ENL 1 s) against drift .86&ndash;.97. '
          'The chronic far-contra pre-cue fall (.58&ndash;.81) is drift-sized (drift .63&ndash;.67), and the acute far-contra pre-cue rise (1.84&ndash;2.21) is beyond drift. '
          'Do not quote near-ipsi pre-cue ratios of 2&ndash;3&times; with 0 bins: they come from a near-zero reference amplitude.</div>')

LAYOUT = [
    ('<h3 class="h7">7.1 PRE-CUE windows</h3>\n' + TAKE71 + '\n<h4 class="h7">7.1.1 PRE-CUE 2 s window, REST reference</h4>', pc),
    ('<h4 class="h7">7.1.2 ENL 1 s window, REST reference</h4>', enl),
    ('<h3 class="h7">7.2 CUE window</h3>\n<h4 class="h7">7.2.1 REST reference</h4>', cue_r),
    ('<h4 class="h7">7.2.2 PRECUE reference</h4>', cue_p),
    ('<h3 class="h7">7.3 LICK window</h3>\n<h4 class="h7">7.3.1 REST reference</h4>', lick_r),
    ('<h4 class="h7">7.3.2 PRECUE reference</h4>', lick_p),
]

# number in page order: 7.<window index><letter>
NUM = {}
win_idx = {"pc": 1, "enl": 1, "cue": 2, "lick": 3}
count = {}
for _, keys in LAYOUT:
    for k in keys:
        w = win_idx[F[k]["w"]]
        count[w] = count.get(w, 0) + 1
        NUM[k] = "7.%d%s" % (w, chr(96 + count[w]))


def fid(k):
    return "fig" + NUM[k].replace(".", "_")


def resolve(t):
    return re.sub(r"\{(\w+)\}", lambda m: NUM[m.group(1)], t)


def render(k):
    d = F[k]
    title = "%s &middot; %s &middot; %s &middot; %s" % (WIN[d["w"]], REF[d["r"]], GATE[d["g"]], KIND[d["k"]])
    alt = title.replace("&middot;", "-").replace("&rarr;", "to")
    return ('<figure id="%s"><img src="%s" alt="%s" loading="lazy"><figcaption><b class="h">Fig %s &middot; %s.</b> %s</figcaption></figure>'
            % (fid(k), d["src"], alt, NUM[k], title, resolve(d["body"])))


def cell(*keys):
    return ", ".join('<a href="#%s">%s</a>' % (fid(k), NUM[k]) for k in keys)


idx = ('<div class="tbl"><table>\n'
       '<tr><th>window &middot; reference</th><th>contact gate</th><th>LP-lick gate</th><th>June &rarr; Aug drift (contact | LP-lick)</th></tr>\n'
       '<tr><td>PRE-CUE 2 s &middot; REST</td><td>%s</td><td>%s</td><td>%s | %s</td></tr>\n'
       '<tr><td>ENL 1 s &middot; REST</td><td>%s</td><td>no twin (lick-free trials)</td><td>%s | &ndash;</td></tr>\n'
       '<tr><td>CUE &middot; REST</td><td>%s, %s per animal</td><td>%s</td><td>%s | %s</td></tr>\n'
       '<tr><td>CUE &middot; PRECUE</td><td>%s</td><td>%s</td><td>%s | %s</td></tr>\n'
       '<tr><td>LICK &middot; REST</td><td>%s</td><td>%s (headline), %s per animal</td><td>%s | %s</td></tr>\n'
       '<tr><td>LICK &middot; PRECUE</td><td>%s</td><td>%s</td><td>%s | %s</td></tr>\n'
       '</table></div>\n<p class="input">Drift index complete (10-25): all three arms (contact, LP-lick, ENL) rendered; summary tables in '
       '<code>grant_figures/map_drift_prepre/&lt;arm&gt;/drift_vs_stroke_panels.csv</code>.</p>') % (
    cell("pcRC"), cell("pcRL"), cell("pcRCd"), cell("pcRLd"),
    cell("enlRC"), cell("enlRCd"),
    cell("cueRC"), cell("cueRCa"), cell("cueRL"), cell("cueRCd"), cell("cueRLd"),
    cell("cuePC"), cell("cuePL"), cell("cuePCd"), cell("cuePLd"),
    cell("lickRC"), cell("lickRL"), cell("lickRLa"), cell("lickRCd"), cell("lickRLd"),
    cell("lickPC"), cell("lickPL"), cell("lickPCd"), cell("lickPLd"))

m = re.search(r'<section id="maps".*?</section>', s, re.S)
old = m.group(0)
i0 = old.index('<div class="tbl">')
i1 = old.index("</div>", old.index('<div class="take">')) + len("</div>")
keep = old[i0:i1]
keep = keep.replace("<th>far contra amplitude / pre</th>", "<th>far contra amplitude / pre (RESTW reference)</th>")
keep = keep.replace("Caveat: the pre-stroke rest SD",
                    "The cue numbers in this table and takeaway come from the position-weighted rest (RESTW) maps; the time-local REST maps used below give .48 / .66 / 1.43 (Fig %s), the same conclusion. Caveat: the pre-stroke rest SD" % NUM["cueRC"])
body = "\n".join(h + "\n" + "\n".join(render(k) for k in keys) for h, keys in LAYOUT)
new = ('<section id="maps" data-toc="7. Maps: far-contra amplitude" data-kind="n">\n'
       '<span class="tag">Anatomy &middot; position maps by window, reference and gate</span>\n'
       '<h2>7. The far-contra cortical response roughly halves acutely and returns <span class="kind n">neural</span><span class="st stands">stands</span></h2>\n'
       '<p class="input">Every map is titled <b>window &middot; reference &middot; gate</b>. Windows, always in this order: PRE-CUE 2 s, ENL 1 s (lick-free pre-cue), CUE (0&ndash;2 s after the cue), LICK. References: REST = each session&rsquo;s time-local rest baseline; PRECUE = each trial&rsquo;s own pre-cue window, so it shows the cue-evoked <i>increment</i> and removes the anticipatory position signal (no PRECUE version of the pre-cue windows, by construction). Gates: contact = DAQ spout-contact trials; LP-lick = Lightning Pose lick-attempt trials. Figures nest window &gt; reference &gt; gate: contact pooled, [per animal], contact drift, then LP-lick pooled, [per animal], LP-lick drift. Drift = the same map in June vs August pre-stroke sessions (no stroke).</p>\n'
       + keep + "\n" + idx + "\n" + body + "\n</section>")
s = s.replace(old, new)

rep("(Fig 7k), so treat it as not separable from drift.", "(Fig %s), so treat it as not separable from drift." % NUM["cueRCd"])
rep("h2 { font:600 1.15rem/1.3",
    "h3.h7 { font:600 1.02rem/1.3 \"Source Serif 4\", Georgia, serif; margin:26px 0 6px; padding-top:8px; border-top:1px solid var(--rule) }\n"
    "h4.h7 { font:600 .8rem/1.3 \"Source Sans 3\", sans-serif; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:16px 0 6px }\n"
    "h2 { font:600 1.15rem/1.3")
rep("var m = /^fig(\\d+)/.exec(f.id); var num = m ? m[1] : f.id;",
    "var m = /^fig(\\d+(?:_\\d+)?)/.exec(f.id); var num = m ? m[1] : f.id;")
p.write_text(s, encoding="utf-8")
print("ok", " ".join("%s=%s" % (k, NUM[k]) for _, ks in LAYOUT for k in ks))
print("files:", " ".join(F[k]["src"] for _, ks in LAYOUT for k in ks))

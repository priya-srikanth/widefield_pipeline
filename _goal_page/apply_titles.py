import csv, html, re
from pathlib import Path
S = Path(r"C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad")
G = S / "goal_page" / "index.html"; s = G.read_text(encoding="utf-8")
T = {r["fid"]: r["title"] for r in csv.DictReader(open(S / "fig_titles.csv", encoding="utf8"))}
T.update({
 "fig1-37": "NEW-design partial refit, target vs landing labels: chronic lick-locked activity follows the target spout more than landing",
 "fig1-38": "OLD-design partial refit of target vs executed-position columns against nulls: chronic remapping in both, not target-only",
 "fig3-7": "Decoding lateral tongue-tip position of ITI and in-trial licks: per-session decoder intact, frozen pre-stroke decoder lower",
 "fig3-13": "Time-binned frozen pre-stroke decoder (all / orofacial / visual+RSP areas): ITI lower everywhere; in-trial recovers except in visual+RSP",
 "fig3-15": "Frozen pre-stroke decoders (Allen areas) for lateral, outward, 2-D, angle and reach: the ITI loss is lateral",
 "fig93l": "Frozen pre-stroke lateral-position decoder in three spatial bases: ITI loss in every basis, in-trial recovers",
})
# 1. new figure after 3.15: lateral position x bases
cap = ('<figure id="fig93l"><img src="iti_where_decoder_bases.png" alt="Lateral-position frozen decoder in three bases"><figcaption><b class="h">Fig 93l &middot; NEURAL &middot; the lateral-position result in three spatial bases.</b> '
 '<b>Decoder:</b> FROZEN pre-stroke (fit once on each animal&rsquo;s August pre licks, never refit). <b>Measured:</b> r between decoded and actual lateral tongue-tip position, per session, divided by the animal&rsquo;s pre mean. '
 'Same licks and decoder as <a href="#fig3-15">3.15</a> (top-left panel), in Allen-area features, each animal&rsquo;s joint LocaNMF basis and its pre-stroke SVD basis. Dots = sessions; black = mean &plusmn; SEM; * = nested-bootstrap change from pre excluding 0. '
 'ITI lateral position stays lower in every basis (chronic &minus;19 to &minus;20%, 4/4 animals each); in-trial lateral position recovers to &minus;4 to &minus;9%. The other targets per basis are in <code>where_decoder_tests.csv</code>. Script <code>scripts/iti_where_decoder.py</code>. PROVISIONAL.</figcaption></figure>\n')
a = '<figure id="fig3-16">'
if "fig93l" not in s:
    assert s.count(a) == 1; s = s.replace(a, cap + a, 1)
# 2. 3.15 caption: now Allen areas only
a = "Rows = basis (Allen areas, joint LocaNMF, pre-stroke SVD basis) &times; lick type;"
if a in s:
    s = s.replace(a, "Allen-area features (the three bases are compared for lateral position in the next figure); rows = lick type, columns = target;", 1)
# 3. decoder / measured lines for section 3 captions
DEC_F = "<b>Decoder:</b> FROZEN pre-stroke (fit once on each animal&rsquo;s August pre licks, never refit)"
MEAS_R = "<b>Measured:</b> r between decoded and actual lateral tongue-tip position"
lines = {
 "fig3-7": "<b>Decoder:</b> left / middle = PER-SESSION (refit within each session, cross-validated); right = FROZEN pre-stroke (fit once on August pre ITI licks, never refit). " + MEAS_R + ".",
 "fig3-8": DEC_F + ". " + MEAS_R + ", per lag from the lick peak.",
 "fig3-9": DEC_F + ". " + MEAS_R + " (lags 0&ndash;0.2 s), one dot per session. <b>ITI vs in-trial, tested directly</b> (same sessions, ITI minus in-trial change from pre): chronic &minus;0.062 [&minus;0.110, &minus;0.014], 3 of 4 animals (PS92 &minus;0.06, PS94 &minus;0.07, PS95 &minus;0.12, PS93 0.00); subacute &minus;0.041 (CI reaches 0). So in-trial licks recover more than ITI licks in three of four animals.",
 "fig3-10": DEC_F + ". " + MEAS_R + " (/ pre), against far-contra hit rate (behaviour).",
 "fig3-11": "<b>No decoder.</b> <b>Measured:</b> behaviour only (ITI licks per minute, contact share).",
 "fig3-12": DEC_F + ", time-binned features. " + MEAS_R + ".",
 "fig3-13": DEC_F + ", time-binned features. " + MEAS_R + ", one number per session; black = mean &plusmn; SEM.",
 "fig3-14": DEC_F + ", three spatial feature bases. " + MEAS_R + ".",
 "fig3-15": DEC_F + ". <b>Measured:</b> r between decoded and actual tongue-tip position (one decoder per target), / animal&rsquo;s pre.",
}
for k in ("fig3-16", "fig3-17", "fig3-19", "fig3-23", "fig3-24"):
    lines[k] = "<b>No decoder.</b> <b>Measured:</b> dF/F around the lick (activity, not information)."
for k in ("fig3-18", "fig3-22", "fig3-25"):
    lines[k] = "<b>No decoder.</b> <b>Measured:</b> dF/F AUC = area under the activity curve (response 0&ndash;0.5 s, build-up &minus;0.4 to &minus;0.05 s); not a decoder r."
for k in ("fig3-20", "fig3-21"):
    lines[k] = "<b>No decoder.</b> <b>Measured:</b> dF/F response / build-up with lick kinematics or face movement regressed out."
for fid, ln in lines.items():
    m = re.search(r'(<figure id="' + fid + r'">.*?<figcaption><b class="h">.*?</b>)', s, re.S)
    assert m, fid
    if "<b>Decoder:</b>" in s[m.end():m.end() + 300] or "<b>No decoder.</b>" in s[m.end():m.end() + 300]:
        continue
    s = s[:m.end()] + " " + ln + s[m.end():]
# 4. titles above each image
css = (".ftitle { font:600 1.02rem/1.35 \"Source Sans 3\", sans-serif; color:var(--fg); margin:0 0 8px }\n"
       ".ftitle .fnum { color:var(--muted); font-weight:600; margin-right:6px }\n")
if ".ftitle" not in s:
    s = s.replace("figcaption { color:var(--muted);", css + "figcaption { color:var(--muted);", 1)
s = re.sub(r'<div class="ftitle">.*?</div>', "", s)
def addt(m):
    fid = m.group(1)
    num = re.search(r'<b class="h">(Fig [0-9a-z.]+|Key figure)', m.group(0))
    lab = num.group(1) if num else ""
    return m.group(0).replace("<img", f'<div class="ftitle"><span class="fnum">{lab}</span>{html.escape(T[fid])}</div><img', 1)
s = re.sub(r'<figure id="([^"]+)">.*?</figure>', addt, s, flags=re.S)
# 5. TOC: use the title
a = r"var txt = b ? b.textContent.replace(/\.$/, '') : (cap ? cap.textContent.slice(0, 38) + '…' : f.id);"
assert a in s, "toc line"
s = s.replace(a, "var ft = f.querySelector('.ftitle');\n        var txt = ft ? ft.textContent : (b ? b.textContent.replace(/\\.$/, '') : (cap ? cap.textContent.slice(0, 38) + '…' : f.id));", 1)
G.write_text(s, encoding="utf-8")
print("figures", s.count('<figure id='), "titles", s.count('class="ftitle"'))

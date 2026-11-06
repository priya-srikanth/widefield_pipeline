"""Section 1 = the REFIT decoder only (10-27; e3 found the mismatch, the PI deck slide 4 was fixed the same way).

Section 1's take and table describe a decoder refit within each session, but its first figure (d_within_precue) is the
FROZEN pre-stroke decoder. This step:
  * moves d_within_precue to the top of section 2 (frozen decoder; it is the 2 s twin of d_enl1s);
  * adds the refit by-position figures (family 5r, precue + cue, working) to section 1 before d_slowband;
  * corrects the table's session counts (acute 5 / 4 / 6 / 1, read from qc/slowband_decode_all.csv; the page said 4/4/4/1)
    and says it is a different refit (Allen-ROI features, engaged trials, August-only pre) from the figures;
  * renumbers sections 1-2 by position (figures keep their stable ids AND their old-number anchors, so #fig1-1 still
    lands on the frozen figure it meant), refreshes every printed number, and rewrites imaging_fig_ids.csv.

Run after p_img_stableids.py. Asserts before writing. PNGs: page_figs.py d_refit_precue=<5r precue> d_refit_cue=<5r cue>.
"""
import csv
import html as H
import re
from pathlib import Path

HERE = Path(__file__).parent
PAGE = HERE / "imaging_page" / "imaging_results.html"
REG = HERE / "imaging_fig_ids.csv"
SRC = "M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_5r_refit_by_position_{}_working.png"

h = PAGE.read_text(encoding="utf8")
assert 'id="f-d_within_precue"' in h and 'id="f-d_refit_precue"' not in h


def sec(sid):
    m = re.search(r'<section id="%s".*?</section>' % sid, h, re.S)
    assert m, sid
    return m


def fig(fid):
    m = re.search(r'<figure id="f-%s".*?</figure>' % fid, h, re.S)
    assert m, fid
    return m.group(0)


EPOCHS = ("Epochs per animal from its own lesion day (acute = far-contra hit rate &lt; 25% of pre: PS92 days 1&ndash;5, "
          "PS93 1&ndash;4, PS94 1&ndash;7, PS95 1; chronic from day 11 / 11 / 25 / 15; subacute between).")
COMMON = (
    "<b>Decoder:</b> WITHIN-SESSION refit 6-way multinomial logistic regression (standardised, C = 0.5) on each "
    "animal&rsquo;s joint-LocaNMF dorsal-cortex components in time sub-bins (session z-scored), fit and scored on each "
    "session by itself with 5-fold block cross-validation (GroupKFold by ~6-trial position blocks); a position with "
    "&lt; 10 trials or &lt; 1/18 of the session is dropped from that session. Same features and estimator as the frozen "
    "decoder of section 2; only the training set differs. <b>Measured:</b> accuracy of the true spout, chance 1/6. "
    "{window} No per-trial baseline. <b>Trials:</b> working trials post-stroke (licking trials plus misses while still "
    "working; terminal quit period excluded), licking trials pre-stroke; no contact or LP gate. Movement not regressed "
    "out. <b>Stats:</b> bars = trial-pooled accuracy with bootstrap 95% CI, dots = sessions coloured by animal; nested "
    "bootstrap (animals &rarr; sessions &rarr; position blocks, 2,000 draws); * = 95% CI of epoch &minus; pre excludes 0 "
    "(uncorrected), ** = still excludes 0 after Bonferroni over 18 tests (6 positions &times; 3 post epochs). 4 animals; "
    "sessions pre 44 (June + August, 11 per animal) / acute 16 / subacute 17 / chronic 43. " + EPOCHS +
    " <b>Not the table above:</b> that is a different refit (Allen-ROI features, engaged trials, August-only pre, plain "
    "session means). <b>Reading:</b> {reading} <b>Script:</b> <code>wfield_local/epoch_accuracy.py</code> "
    "(<code>_frozen_vs_refit</code>) on <code>grant_confusion._refit_pred</code>; epoch family 5r, rendered 10-07."
)
NEW = {
    "d_refit_precue": (
        "Within-session refit spout decoder, pre-cue 2 s window: position decodable in every epoch, no acute loss at "
        "any position",
        "Pre-cue decoding by position, refit within each session",
        "pre-cue (ENL) decoding by position and epoch, decoder refit within each session",
        "<b>Window:</b> 2 s ending at the cue (4 &times; 0.5 s bins); if a DAQ lick falls inside, the window slides back "
        "to the latest lick-free 2 s gap after the spout arrives, and trials with none are dropped.",
        "No position loses pre-cue accuracy acutely; the only marked change is a chronic rise at near middle (**). Far "
        "contra sits at about 0.3&ndash;0.4 in every epoch, so a low pre-cue far-contra code is not specific to the frozen "
        "decoder.",
    ),
    "d_refit_cue": (
        "Within-session refit spout decoder, cue window: accuracy falls acutely at every position, most at the far "
        "spouts, and is back by chronic",
        "Cue-window decoding by position, refit within each session",
        "cue-window decoding by position and epoch, decoder refit within each session",
        "<b>Window:</b> cue to cue + 2 s (4 &times; 0.5 s bins).",
        "Every position falls acutely (near contra and the three far positions after Bonferroni): far contra about "
        "0.82 &rarr; 0.43, far middle 0.76 &rarr; 0.40, far ipsi 0.80 &rarr; 0.46 (read off the bars). By chronic every "
        "position is at or above pre. A decoder trained on the acute session itself still loses the far spouts, so the "
        "acute post-cue loss of section 2 is not only a frozen-decoder mismatch.",
    ),
}


def newfig(fid):
    title, alt, head, window, reading = NEW[fid]
    cap = COMMON.format(window=window, reading=reading)
    return (f'<figure id="f-{fid}" data-fid="{fid}"><div class="ftitle"><span class="fnum">Fig 0.0</span>{title}</div>'
            f'<img src="{fid}.png" alt="{alt}" loading="lazy"><figcaption><b class="h">Fig 0.0 &middot; {head}.</b> '
            f"{cap}</figcaption></figure>")


# 1. move the frozen pre-cue figure to the top of section 2
frozen = fig("d_within_precue")
h = h.replace(frozen + "\n", "", 1) if frozen + "\n" in h else h.replace(frozen, "", 1)
enl = fig("d_enl1s")
h = h.replace(enl, frozen + "\n" + enl, 1)
# 2. refit figures into section 1, before the slow-band figure
slow = fig("d_slowband")
h = h.replace(slow, newfig("d_refit_precue") + "\n" + newfig("d_refit_cue") + "\n" + slow, 1)

# 3. section 1 table header + input + take
old_hdr = "<h4>Per animal (session means; Allen-ROI features; pre n = 8, acute 4 / 4 / 4 / 1)</h4>"
assert old_hdr in h
h = h.replace(old_hdr, "<h4>Per animal (session means; Allen-ROI features, engaged trials; sessions pre 8 per animal "
              "(August), acute 5 / 4 / 6 / 1, subacute 2 / 3 / 5 / 7, chronic 12 / 12 / 8 / 11)</h4>", 1)
old_in = "Multinomial logistic on cortex features, block cross-validation, chance 0.167.</p>"
assert old_in in h
h = h.replace(old_in, "Multinomial logistic on cortex features, <b>refit on each session</b> (block cross-validation), "
              "chance 0.167. The frozen pre-stroke decoder is section 2.</p>", 1)
old_take = ("against chance 0.167. There is no consistent acute pre-cue loss (up in PS92 / PS95, down in PS93 / PS94).")
assert old_take in h
h = h.replace(old_take, (
    "against chance 0.167 (table). By position (Figs 1.1&ndash;1.2, joint-LocaNMF features): pre-cue, no position loses "
    "accuracy acutely; post-cue, a decoder refit on the acute session itself still loses every position, most at the far "
    "spouts (far contra about 0.82 &rarr; 0.43), and is back by chronic. There is no consistent acute pre-cue loss per "
    "animal either (up in PS92 / PS95, down in PS93 / PS94)."), 1)

# 4. renumber sections 1 and 2 by position; record old -> new for printed numbers
renum = {}
for sid, n in (("within", 1), ("frozen", 2)):
    body = sec(sid).group(0)
    new_body = body
    for k, fid in enumerate(re.findall(r'<figure id="f-([^"]+)"', body), 1):
        f0 = re.search(r'<figure id="f-%s".*?</figure>' % fid, new_body, re.S).group(0)
        old = re.search(r'<span class="fnum">Fig ([\d.]+)</span>', f0).group(1)
        new = f"{n}.{k}"
        renum[fid] = (old, new)
        f1 = f0.replace(f'<span class="fnum">Fig {old}</span>', f'<span class="fnum">Fig {new}</span>', 1)
        f1 = f1.replace(f'<b class="h">Fig {old} &middot;', f'<b class="h">Fig {new} &middot;', 1)
        assert f1.count(f"Fig {new}") >= 2, fid
        new_body = new_body.replace(f0, f1, 1)
    h = h.replace(body, new_body, 1)
print({k: v for k, v in renum.items()})
# printed numbers next to stable links
for fid, (_old, new) in renum.items():
    h = re.sub(r'(<a href="#f-%s">)[\d.]+(</a>)' % re.escape(fid), r"\g<1>%s\g<2>" % new, h)
# plain-text refs (checked by hand: the only two outside links)
for a, b in (("lick gate for the lick window (Fig 2.3).", "lick gate for the lick window (Fig 2.4)."),
             ("Position = the Fig 1.1 decoder", "Position = the Fig 2.1 decoder")):
    assert h.count(a) == 1, a
    h = h.replace(a, b, 1)
# where-things-stand rows
row1 = '<a href="#f-d_within_precue">2.1</a>, <a href="#f-d_slowband">1.3</a>, <a href="#within">&sect; table</a>'
assert row1 in h
h = h.replace(row1, '<a href="#f-d_refit_precue">1.1</a>, <a href="#f-d_refit_cue">1.2</a>, '
              '<a href="#f-d_slowband">1.3</a>, <a href="#within">&sect; table</a>', 1)
row2 = '<a href="#f-d_acc_cue">2.3</a>, <a href="#f-d_lp_lick">2.4</a>, <a href="#f-d_enl1s">2.2</a>'
assert row2 in h
h = h.replace(row2, row2 + ', <a href="#f-d_within_precue">2.1</a>', 1)

PAGE.write_text(h, encoding="utf8")

# 5. registry (fid -> current number), same format as p_img_stableids
rows = []
for m in re.finditer(r'<figure id="f-([^"]+)"[^>]*>(.*?)</figure>', h, re.S):
    fid, body = m.groups()
    num = re.search(r'<span class="fnum">Fig ([\d.]+)</span>', body).group(1)
    title = re.sub(r"<[^>]+>", "", re.search(r'<span class="fnum">[^<]*</span>(.*?)</div>', body, re.S).group(1))
    rows.append((fid, num, H.unescape(title).strip()))
assert len(rows) == 76 and len({r[1] for r in rows}) == 76
with REG.open("w", newline="", encoding="utf8") as f:
    w = csv.writer(f)
    w.writerow(["fid", "number", "title"])
    w.writerows(rows)
print(f"{len(rows)} figures; registry rewritten")

import re, sys

SRC = r"C:/Users/SabatiniLab/AppData/Local/Temp/claude/179e1046909c5be2/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/s23_page.html"
OUT = SRC.replace("s23_page.html", "s23_check_" + sys.argv[1].split("/")[-1].replace(".py","") + ".html")
s = open(SRC, encoding="utf8").read()
n0 = len(s)


def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, (c, old[:120])
    s = s.replace(old, new)


RUN = ("<code>rebuild_20261103/labcams/&lt;tag&gt;/cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118/</code> "
       "(whole chain re-extracted with the last-sync end, code 871be8f9, runner <code>quiet_c1guard/run_chain_syncend.sh</code>; "
       "old vs new numbers: <code>docs/status/quiet_restyle_20261118/QUIET_SYNCEND_OLD_NEW_20261118.md</code>)")
QF = ("<b>Quiet frames:</b> engaged part of the session &mdash; before the quit and before the last sync pulse "
      "(<code>QS_ENGAGED_END=lastsync</code>; the task controller stops 0&ndash;25 s after the last cue, so imaging after "
      "the task is not counted; Priya 11-18) &mdash; with no lick and no running bout within 1.5 s before or after (spout may be out).")
LK = ("<b>Licks:</b> DAQ spout contact OR LP video lick, whole session: cam4 <code>licks_all</code> plus cam1-only events that are "
      "non-droplet with med_out &ge; &minus;120 px (cam1-only labels far from the mouth are not counted); PS93 08-13 has no behavior "
      "video, so its quiet frames use DAQ licks only.")
TH = "<b>Threshold:</b> k = 4 &times; the animal&rsquo;s own August-pre robust SD (1.4826 &times; MAD). <b>Baseline:</b> each animal&rsquo;s own August pre."
SESS = "108 sessions (pre 32, 8 per animal; acute 16, subacute 17, chronic 43)"
STARS = ("Open markers = sessions minus their animal&rsquo;s own pre mean (shape = animal, color = epoch); filled = animal-weighted "
         "change with its 95% CI; <b>*</b> = nested animal &rarr; session bootstrap 95% CI of the change excludes 0 (primary); "
         "&dagger; = mixed model q &lt; .05 (backup, FDR within measure); k/4 = animals changing in the pooled direction (top band).")


def data_block(extra=""):
    return (f"<b>Data:</b> rebuilt base with the cam1 guard, analyzed time ending at the last sync pulse, {RUN}; {SESS}. "
            f"{QF} {LK} {TH}{extra}")


def set_caption(fid, cap):
    global s
    a = s.index(f'<figure id="f-{fid}"')
    b = s.index("<figcaption>", a) + len("<figcaption>")
    c = s.index("</figcaption>", b)
    s = s[:b] + cap + s[c:]


def set_img(old, new):
    rep(f'<img src="{old}"', f'<img src="{new}"')


def set_title(fid, old, new, short=None):
    """ftitle text (after the [licks...] tag), alt, summary line, data-short."""
    global s
    a = s.index(f'<figure id="f-{fid}"')
    e = s.index("</figure>", a)
    seg = s[a:e]
    assert seg.count(old) >= 2, (fid, seg.count(old))
    seg = seg.replace(old, new)
    if short:
        seg = re.sub(r'data-short="[^"]*"', f'data-short="{short}"', seg, count=1)
    s = s[:a] + seg + s[e:]
    d = s.rfind("<details", 0, a)
    if d != -1 and s.find("</summary>", d) < a and f'data-for="f-{fid}"' in s[d:a]:
        se = s.index("</summary>", d)
        s = s[:d] + s[d:se].replace(old, new) + s[se:]


exec(open(sys.argv[1], encoding="utf8").read())
open(OUT, "w", encoding="utf8").write(s)
print("ok", n0, len(s), OUT)

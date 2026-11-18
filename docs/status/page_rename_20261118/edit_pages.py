"""Page rename + reorganization (Priya 11-18): spout (target) page vs movement (execution) page.
Usage: python edit_pages.py <imaging_in> <imaging_out> <encoding_in> <encoding_out>
Every replacement must hit exactly once (asserted)."""
import sys

IMG = "https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV"
ENC = "https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d"


def rep(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, old[:120])
    return s.replace(old, new)


def edit_imaging(s):
    s = rep(s, "<title>Cortical Spout Coding After Stroke</title>", "<title>Spout (Target) Coding After Stroke</title>")
    s = rep(s, "<h1>Where the cortical spout code goes after a striatal stroke</h1>",
            "<h1>Spout (target) coding after a striatal stroke: where the cortical spout code goes</h1>\n"
            '<p class="sub" id="pagerole"><b>This page:</b> whether and where cortex carries the TARGET spout &mdash; '
            "position decoders, the frozen pre-stroke decoder, specificity, maps, coding directions, representational "
            "distance (RSA), the pre-cue code and rest. What cortex carries about the EXECUTED movement (encoders, the "
            "tongue / lick read-out, FaceRhythm, behavioral state, quiet-period events) is on the companion page, "
            f'<a href="{ENC}">Movement (Execution) Coding After Stroke</a>.</p>')
    # scope sentence: kinematic regressors and the quiet events now live on the movement page
    s = rep(s, "no kinematic regressors except section 15&rsquo;s face-movement regression (those are on the encoding window's page;",
            f'no kinematic regressors (those, and the quiet-period events that were section 15, are on the <a href="{ENC}">movement (execution) page</a>;')
    # status-table link text
    s = rep(s, f'<a href="{ENC}#quiet-state">encoding page section 23</a>',
            f'<a href="{ENC}#quiet-state">movement page section 23</a>')
    # captions that name the old page as the source (section 5)
    s = rep(s, "<b>Reading:</b> (from the encoding window's page).", "<b>Reading:</b> (from the movement page).")
    s = rep(s, "(encoding window&rsquo;s page script)", "(movement page script)", n=2)
    # section 14 Haufe note
    s = rep(s, "<b>Not the same maps as the encoding page&rsquo;s Haufe figures</b>",
            "<b>Not the same maps as the movement page&rsquo;s Haufe figures</b>")
    # pending list
    s = rep(s, "statistics on the trial-class maps (encoding window).</li>",
            "statistics on the trial-class maps (movement page).</li>")

    # section 5 / 8 cross-references (inserted after each section's takeaway)
    t5 = ("Not yet separable from the failed movement itself.</div>\n")
    s = rep(s, t5, t5 + f'<p class="input"><b>See also:</b> the movement page shows the same failed attempts from the '
            f'executed-movement side, with pooled retention, per-session values and a class-matched pre-cue test '
            f'(<a href="{ENC}#decoder">movement page section 8</a>).</p>\n')
    t8 = ("Ipsilesional ALM reduced more than contralesional in 4 / 4. No statistics yet.</div>\n")
    s = rep(s, t8, t8 + f'<p class="input"><b>See also:</b> the movement page shows the failed-attempt dF/F maps '
            f'and amplitudes per trial type (<a href="{ENC}#maps">movement page section 9</a>).</p>\n')

    # section 15 -> short pointer; keep every id / anchor so old links land
    a = s.index('<section id="quiet" data-toc="15. Quiet-period events" data-kind="n">')
    b = s.index("</section>", a) + len("</section>")
    old = s[a:b]
    for fid in ["f-q_fr", "f-q_examples", "f-q_fr_detail", "f-q_threshold", "f-q_spectrum", "f-q_sdmaps", "f-q_eta_maps"]:
        assert f'id="{fid}"' in old, fid
    assert old.count(' id="') == 8, old.count(' id="')  # section + 7 figures
    anchors = "".join(f'<span class="oldnum" id="{fid}"></span>' for fid in
                      ["f-q_fr", "f-q_examples", "f-q_fr_detail", "f-q_threshold", "f-q_spectrum", "f-q_sdmaps", "f-q_eta_maps"])
    new = ('<section id="quiet" data-toc="15. Quiet-period events (on the movement page)" data-kind="n">\n'
           '<span class="tag">Spontaneous activity &middot; quiet periods &middot; moved to the movement page</span>\n'
           + anchors + "\n"
           '<h2>15. Quiet-period events: on the movement page <span class="kind n">neural</span></h2>\n'
           '<p>The quiet-period analysis (brief orofacial-cortex events and the 2&ndash;3 Hz fluctuation during quiet '
           'periods, after regressing out face movement) is about cortex outside the task and uses face-movement '
           'regressors, so it lives on the movement page, which holds the current results (the former Figs '
           '15.1&ndash;15.7 of this section are superseded there): '
           f'<a href="{ENC}#quiet-state">movement page section 23, Quiet-period cortical events</a>.</p>\n'
           "</section>")
    s = s[:a] + new + s[b:]
    return s


def edit_encoding(s):
    s = rep(s, "<title>Attempt Encoding Figures</title>", "<title>Movement (Execution) Coding After Stroke</title>")
    s = rep(s, "<h1>What cortex looks like when a post-stroke attempt fails</h1>",
            "<h1>What cortex carries about the executed movement after a striatal stroke: lick, tongue, face and state</h1>\n"
            '<p class="sub" id="pagerole"><b>This page:</b> what cortex carries about the EXECUTED movement &mdash; '
            "encoders, the tongue / lick read-out, FaceRhythm, behavioral state and quiet-period events. Whether and "
            "where cortex carries the TARGET spout (position decoders, frozen decoder, specificity, maps, coding "
            "directions, RSA, pre-cue code, rest) is on the companion page, "
            f'<a href="{IMG}">Spout (Target) Coding After Stroke</a>.</p>')
    s = rep(s, "the pre-stroke spatial template&rsquo;s shape stays off 4/4 (imaging page, FINDINGS 3.12).",
            f'the pre-stroke spatial template&rsquo;s shape stays off 4/4 (<a href="{IMG}#encoder">spout (target) page section 6</a>, FINDINGS 3.12).')
    s = rep(s, "moved from the imaging page section 15 (11-01)</span>",
            "moved from the spout (target) page section 15 (11-01)</span>")
    s = rep(s, "Moved here from the imaging page (Priya 11-01).",
            f'Moved here from the <a href="{IMG}#quiet">spout (target) page</a> (Priya 11-01).')
    t8 = ("Failed attempts stay low in every epoch; successful hits recover but stay below pre even chronically.</div>\n")
    s = rep(s, t8, t8 + f'<p><b>See also:</b> the spout (target) page shows the same trial-class decoder with per-animal '
            f'numbers, under the target-coding question (<a href="{IMG}#attempts">spout (target) page section 5</a>).</p>\n')
    t9 = ("chronic hits <b>increase tjM1 / SSp-m</b>, more contralesionally, at every spout (4/4).</div>\n")
    s = rep(s, t9, t9 + f'<p><b>See also:</b> the spout (target) page shows the rest-referenced (REST and RESTnl) '
            f'far-contra attempt-class maps with per-animal ratios (<a href="{IMG}#attmaps">spout (target) page section 8</a>).</p>\n')
    return s


if __name__ == "__main__":
    ii, io, ei, eo = sys.argv[1:5]
    for fi, fo, fn in [(ii, io, edit_imaging), (ei, eo, edit_encoding)]:
        s = open(fi, encoding="utf-8", newline="").read()
        open(fo, "w", encoding="utf-8", newline="").write(fn(s))
        print(fo, len(s), "->", len(fn(s)))

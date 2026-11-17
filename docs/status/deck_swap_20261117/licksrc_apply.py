"""PI-deck lick-source wording fixes from 3c's audit (docs/status/LICK_SOURCE_AUDIT_20261117.md section 5): 6 WRONG + the
VAGUE additions. Substring replacement inside notes paragraphs and slide text (current deck text, not the 13:35
snapshot); "add" items append to the matching paragraph or add a line. RESTnl map slides are NOT in this pass.
usage: python licksrc_apply.py <deck_in> <deck_out> <log_csv>   (dry-run = write to a scratch copy)"""
import copy
import csv
import sys

from pptx import Presentation
from pptx.text.text import _Paragraph

C4 = "cam4; cam1 is scored only from the cue to the trial stop"
FIX = [  # (slides, old substring, new substring)
    ([26, 186, 206], "no LP lick (cam4 + cam1)", f"no LP lick ({C4})"),
    ([7], "LP video licks (cam4 LP v36 + cam1, triangulated 3-D tongue)",
     "LP video licks, cam4 only (3-D position from cam1 + cam4 pose)"),
    ([117, 118], "LP cam4 + cam1 3-D licks + DAQ contact",
     "LP video licks, cam4 only, 3-D landing from cam1 + cam4 pose; + DAQ contact"),
    ([15, 16], "LP video licks (WIDEFIELD_LICK_SOURCE=lp; lick-free window and lick column on the video train)",
     "LP video licks (WIDEFIELD_LICK_SOURCE=lp): lick column = LP video, cam4 + cam1 (cam1 droplets removed); pre-cue "
     f"lick-free window = LP video, {C4}"),
    ([57], "LP video licks (cam4 LP v36) + cam1 (3-D)", "LP video licks, cam4 only (3-D position from cam1 + cam4 pose)"),
    ([73], "Licks: none (factors named afterwards against LP licks)",
     "Licks: none in the fit (factors named afterwards against LP video licks, cam4 only)"),
    ([74], "Licks: none (names validated against LP licks / running)",
     "Licks: none in the fit (names validated against LP video licks, cam4 only, and running)"),
    ([77], "Licks: none for the peak (per-lick text: LP licks)",
     "Licks: none for the peak (per-lick text: LP video licks, cam4 only)"),
    ([82, 83, 87], "Licks: LP licks", "Licks: LP video, cam4 only (licks_all.csv; lick and bout onsets)"),
    ([84], "Licks: LP (video) licks", "Licks: LP video, cam4 only (licks_all.csv)"),
    ([88, 90, 91, 92, 93, 95], "Licks: LP (video) licks define the lick factors' context; no DAQ lick gate",
     "Licks: LP video, cam4 only (licks_all.csv) define the lick factors' context; no DAQ lick gate"),
    ([96], "Licks: LP (video) licks (FR lick window); 3-D = camera licks",
     "Licks: FR lick window = LP video, cam4 only (licks_all.csv); 3-D = cam4 LP licks (3-D position from cam1 + cam4 pose)"),
    ([99, 102, 147, 148, 149, 196, 197], "adds own pre-cue (ENL) + ITI LP lick kernels",
     "adds own pre-cue (ENL) + ITI LP lick kernels (cam4 only: the cam1 events are post-cue)"),
    ([102], "adds own pre-cue(ENL) + ITI LP lick kernels",
     "adds own pre-cue (ENL) + ITI LP lick kernels (cam4 only: the cam1 events are post-cue)"),
    ([100], "(own pre-cue + ITI lick kernels)", "(own pre-cue + ITI lick kernels, cam4 only)"),
    ([196], "LP cam4 + cam1 video licks (3-D landing) + DAQ contact",
     "LP cam4 + cam1 video licks (3-D landing on cam4 licks) + DAQ contact"),
]
ADD = [  # (slide, anchor substring in a paragraph or None, text appended / new line)
    (14, "Licks: LP video licks (cam4 + cam1)", f"; pre-cue lick-free window: LP video, {C4}"),
    (153, None, "Licks: DAQ spout contact + LP video licks (cam4 + cam1, cam1 droplets removed) define non-quiet frames "
                "(QS_LICK_SOURCE=daq+lp)."),
]


def paras(slide):
    out = []
    for sh in slide.shapes:
        if sh.has_text_frame:
            out += [(f"slide:{sh.name}", sh.text_frame, p) for p in sh.text_frame.paragraphs]
    if slide.has_notes_slide:
        tf = slide.notes_slide.notes_text_frame
        out += [("notes", tf, p) for p in tf.paragraphs]
    return out


def set_text(p, t):
    if p.runs:
        p.runs[0].text = t
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.text = t


P = Presentation(sys.argv[1])
log = []
for slides, old, new in FIX:
    for n in slides:
        hits = 0
        for _where, _tf, p in paras(P.slides[n - 1]):
            if old in p.text and new not in p.text:
                set_text(p, p.text.replace(old, new))
                hits += 1
        log.append({"slide": n, "kind": "replace", "old": old[:70], "result": f"{hits} paragraph(s)" if hits else "NOT FOUND"})
for n, anchor, text in ADD:
    s = P.slides[n - 1]
    if anchor:
        ps = [p for w, tf, p in paras(s) if anchor in p.text and text not in p.text]
        for p in ps:
            set_text(p, p.text.replace(anchor, anchor + text))
        log.append({"slide": n, "kind": "append", "old": anchor[:70], "result": f"{len(ps)} paragraph(s)" if ps else "NOT FOUND"})
    else:
        tf = s.notes_slide.notes_text_frame
        if any(text in p.text for p in tf.paragraphs):
            log.append({"slide": n, "kind": "add-line", "old": "", "result": "already present"})
            continue
        last = tf.paragraphs[-1]._p
        q = copy.deepcopy(last)
        last.addnext(q)
        set_text(_Paragraph(q, tf), text)
        log.append({"slide": n, "kind": "add-line", "old": "", "result": "added to notes"})
P.save(sys.argv[2])
with open(sys.argv[3], "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["slide", "kind", "old", "result"])
    w.writeheader()
    w.writerows(log)
for r in log:
    print(r["slide"], r["kind"], r["result"], "|", r["old"])

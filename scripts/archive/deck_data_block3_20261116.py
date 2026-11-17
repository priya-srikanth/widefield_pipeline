"""Data: blocks from the per-figure sweep (11-16) for slides whose image is NOT stale. Correction line ends with the
provenance level: [traced: render log] or [traced: code + output file, no run log]."""
import copy, csv, json, os, shutil, sys
from pptx import Presentation
from pptx.text.text import _Paragraph
F = "M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx"
SW = "C:/Users/SabatiniLab/AppData/Local/Temp/claude/179e1046909c5be2/782a4a7a-7855-40ed-b449-494e8aa4b6bc/scratchpad/sweep/sweep_all.json"
LOG = sys.argv[1]
LABELS = ("Window", "Licks", "Trial period", "Trial types", "Gate", "Baseline", "Correction")
KEYS = ("window", "licks", "trial_period", "trial_types", "gate", "baseline", "correction")
TAG = {"LOG": " [traced: render log]", "CODE": " [traced: code + output file, no run log]"}
B = F.replace(".pptx", ".bak_pre_a8_datablock3_20261116.pptx")
if not os.path.exists(B):
    shutil.copy2(F, B)
todo = {o["slide"]: o for o in json.load(open(SW, encoding="utf-8")) if o["stale"] == "no" and o["level"] in TAG}
P = Presentation(F)
log = []
for i, s in enumerate(P.slides, 1):
    if i not in todo:
        continue
    o = todo[i]
    tf = s.notes_slide.notes_text_frame
    title = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame), "")
    norm = lambda t: "".join(c for c in t.lower() if c.isalnum())[:18]
    assert norm(title) == norm(o["title"]), (i, title, o["title"])      # slide numbering unchanged
    head = [p.text for p in tf.paragraphs[:8]]
    if sum(any(h.startswith(l + ":") for h in head) for l in LABELS) == 7:
        continue
    vals = [str(o[k]).strip() or "n/a" for k in KEYS]
    vals[-1] += TAG[o["level"]]
    first = tf.paragraphs[0]._p
    for lab, v in zip(LABELS, vals):
        p = copy.deepcopy(first)
        for r in list(p):
            if r.tag.split("}")[1] in ("r", "br", "fld"):
                p.remove(r)
        first.addprevious(p)
        _Paragraph(p, tf).text = f"{lab}: {v}"
    log.append({"slide": i, "title": title[:80], "level": o["level"], "producing_script": o["producing_script"][:120]})
P.save(F)
with open(LOG, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(log[0]))
    w.writeheader(); w.writerows(log)
print(len(log), "slides written")

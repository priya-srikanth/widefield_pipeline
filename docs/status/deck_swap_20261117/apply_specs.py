"""Apply per-slide swap specs (README.md in this folder) to the PI deck. a8 only; serial.

usage: python apply_specs.py <deck_in> <deck_out> <log_csv> specs1.json [specs2.json ...] [--only 4,6,8]
Applies status == "ready" only. Per slide: assert the title, replace picture blobs in place (fit the new image inside the
old box, centered, crop reset), rewrite / delete slide paragraphs (exact text match), and replace the notes with the
7-line Data: block + the new body. Any mismatch -> that slide is skipped and logged; nothing half-applied."""
import copy
import csv
import json
import re
import sys

from PIL import Image
from pptx import Presentation

LABELS = ("Window", "Licks", "Trial period", "Trial types", "Gate", "Baseline", "Correction")


def norm(s):
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def find_shape(slide, name):
    m = [sh for sh in slide.shapes if sh.name == name]
    if len(m) != 1:
        raise ValueError(f"shape {name!r}: {len(m)} matches")
    return m[0]


def set_para_text(p, text):
    runs = p.runs
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.text = text


def plan_text(slide, edits):
    ops = []
    for e in edits:
        sh = find_shape(slide, e["shape_name"])
        ps = [p for p in sh.text_frame.paragraphs if norm(p.text) == norm(e["old"])]
        if len(ps) != 1:
            raise ValueError(f"text in {e['shape_name']!r} not found once: {e['old'][:60]!r} ({len(ps)})")
        ops.append((ps[0], e["new"]))
    return ops


def swap_picture(slide, pic, path, box=None):
    _, rid = slide.part.get_or_add_image_part(path)
    pic._element.blipFill.blip.rEmbed = rid
    pic.crop_left = pic.crop_top = pic.crop_right = pic.crop_bottom = 0
    W, H = Image.open(path).size
    if box:  # spec override [left, top, width, height] in EMU (layout change, e.g. a much wider figure)
        pic.left, pic.top, pic.width, pic.height = box
    l, t, w, h = pic.left, pic.top, pic.width, pic.height
    a = W / H
    if a > w / h:
        nh = int(w / a)
        pic.width, pic.height, pic.top = w, nh, t + (h - nh) // 2
    else:
        nw = int(h * a)
        pic.width, pic.height, pic.left = nw, h, l + (w - nw) // 2


def write_notes(slide, block, body):
    tf = slide.notes_slide.notes_text_frame
    lines = [f"{k}: {block[k]}" for k in LABELS] + [b for b in body]
    paras = tf.paragraphs
    first = paras[0]._p
    for p in paras[1:]:
        p._p.getparent().remove(p._p)
    from pptx.text.text import _Paragraph
    set_para_text(_Paragraph(first, tf), lines[0])
    prev = first
    for ln in lines[1:]:
        p = copy.deepcopy(first)
        prev.addnext(p)
        set_para_text(_Paragraph(p, tf), ln)
        prev = p


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--only")]
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only"):
            only = {int(x) for x in a.split("=", 1)[1].split(",")}
    deck_in, deck_out, log_csv, spec_files = args[0], args[1], args[2], args[3:]
    specs = [s for f in spec_files for s in json.load(open(f, encoding="utf-8"))]
    P = Presentation(deck_in)
    log = []
    for s in sorted(specs, key=lambda x: x["slide"]):
        n = s["slide"]
        if only and n not in only:
            continue
        if s.get("status") != "ready":
            log.append({"slide": n, "result": f"not applied ({s.get('status')}): {s.get('skip_reason', '')}"})
            continue
        slide = P.slides[n - 1]
        try:
            title = next((sh.text_frame.text for sh in slide.shapes if sh.has_text_frame), "")
            if norm(s["title_now"])[:40] not in norm(title):
                raise ValueError(f"title mismatch: {title[:60]!r}")
            pics = [(find_shape(slide, p["shape_name"]), p["new_image"], p.get("box_emu")) for p in s.get("pictures", [])]
            for _, path, _b in pics:
                Image.open(path).verify()
            ops = plan_text(slide, s.get("slide_text", []))
            blk = s.get("data_block") or {}
            if blk and set(blk) != set(LABELS):
                raise ValueError(f"data_block labels {sorted(blk)}")
        except Exception as e:  # noqa: BLE001
            log.append({"slide": n, "result": f"SKIPPED: {e}"})
            continue
        for pic, path, box in pics:
            swap_picture(slide, pic, path, box)
        for p, new in ops:
            if new == "":
                p._p.getparent().remove(p._p)
            else:
                set_para_text(p, new)
        if blk:
            write_notes(slide, blk, s.get("notes_body", []))
        log.append({"slide": n, "result": f"applied: {len(pics)} picture(s), {len(ops)} text edit(s), "
                                          f"notes {'rewritten' if blk else 'unchanged'}",
                    "conclusion_changed": s.get("conclusion_changed"),
                    "changes": json.dumps(s.get("changes", []), ensure_ascii=False)})
    P.save(deck_out)
    with open(log_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["slide", "result", "conclusion_changed", "changes"])
        w.writeheader()
        w.writerows(log)
    for r in log:
        print(r["slide"], r["result"])


if __name__ == "__main__":
    main()

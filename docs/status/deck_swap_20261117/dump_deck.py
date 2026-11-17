"""Back up the PI deck, then dump every stale slide's text / notes / pictures (name, box, crop, image size) to JSON
for the 11-17 swap pass (Priya 11-17: swap A+B, render C, replace held slides with updated versions)."""
import json, os, shutil, sys
import pandas as pd
from pptx import Presentation
from PIL import Image
import io
F = "M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx"
B = F.replace(".pptx", ".bak_pre_a8_swap_20261117.pptx")
if not os.path.exists(B):
    shutil.copy2(F, B)
st = pd.read_csv("docs/status/DECK_STALE_FIGURES_2026-11-16.csv")
P = Presentation(F)
out = {}
for _, r in st.iterrows():
    s = P.slides[int(r.slide) - 1]
    shapes = []
    for sh in s.shapes:
        d = {"name": sh.name, "type": int(sh.shape_type) if sh.shape_type else None,
             "box_emu": [sh.left, sh.top, sh.width, sh.height]}
        if sh.shape_type == 13:
            im = Image.open(io.BytesIO(sh.image.blob))
            d.update(image_px=list(im.size), crop=[sh.crop_left, sh.crop_top, sh.crop_right, sh.crop_bottom])
        if sh.has_text_frame:
            d["paragraphs"] = [p.text for p in sh.text_frame.paragraphs]
        shapes.append(d)
    notes = [p.text for p in s.notes_slide.notes_text_frame.paragraphs] if s.has_notes_slide else []
    out[int(r.slide)] = {"stale_row": {k: (None if pd.isna(v) else v) for k, v in r.items()},
                         "hidden": s._element.get("show") == "0", "shapes": shapes, "notes": notes}
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, default=str)
print(len(out), "slides dumped; backup", B, os.path.getsize(B))

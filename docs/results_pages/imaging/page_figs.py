"""Copy + downscale the results-page figures into page/ (max 1600 px wide, PNG optimised). Usage: page_figs.py name=path ..."""
import sys
from pathlib import Path

from PIL import Image

out = Path(__file__).with_name("imaging_page")
out.mkdir(exist_ok=True)
for arg in sys.argv[1:]:
    name, src = arg.split("=", 1)
    im = Image.open(src)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    dst = out / f"{name}.png"
    im.save(dst, optimize=True)
    print(f"{name}: {im.width}x{im.height} {dst.stat().st_size // 1024} KB")

"""Write prov_table.tsv: fid, section, page png, source png, current caption (plain text) for every imaging-page figure."""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
s = (HERE / "imaging_page" / "imaging_results.html").read_text(encoding="utf8")
src = {}
for line in (HERE / "fig_sources.txt").read_text().splitlines():
    n, p = line.split(" ", 1)
    src[n] = p
GF = "M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures"
for n in list(src):
    if src[n] != "?":
        continue
    m = re.match(r"(map|drift)_(REST|PRECUE)_(precue|enl1s|cue|lick)_(contact|LP)(_by_animal)?$", n)
    if m:
        kind, ref, w, g, ba = m.groups()
        src[n] = f"(section-7 map; see scratchpad/p_img_s7v3.py; arm {g}; grant_figures/map_drift_prepre/ for drift; epoch_15r*{ref}ref* for maps)"
    elif n.startswith("cp_"):
        src[n] = "(pooled CD-LOO; drawn by scratchpad/p_img_9pooled.py / p_cdloo.py from cd_window_projection_loo_sessions_long.csv)"
    elif n == "m_trend_beh_contact":
        src[n] = "M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/mixed_models/ (contact-deficit trend; written by wfield_local/imaging_lmm.py / mixed_models.py)"
marks = [(m.start(), "S", re.sub("<[^>]+>", "", m.group(0))) for m in re.finditer(r"<h2[^>]*>.*?</h2>", s)]
marks += [(m.start(), "F", m.group(0)) for m in re.finditer(r'<figure id="[^"]+">.*?</figure>', s, re.S)]
marks.sort()
rows, sec = [], ""
for _, k, t in marks:
    if k == "S":
        sec = html.unescape(t)
        continue
    fid = re.search(r'id="([^"]+)"', t).group(1)
    img = re.search(r'src="([^"]+)"', t).group(1)
    cap = html.unescape(re.sub("<[^>]+>", "", t.split("<figcaption>")[1]))
    rows.append("\t".join([fid, sec[:60], img, src.get(Path(img).stem, "?"), " ".join(cap.split())]))
(HERE / "prov_table.tsv").write_text("fid\tsection\tpage_png\tsource\tcaption\n" + "\n".join(rows) + "\n", encoding="utf8")
print(len(rows))

"""CLI for the stroke-extent schematic. Cohort data: configs/histology_map/<cohort>/ (cohort.yaml, sections.json, masks/).

    # 1. where the scans are (Mac / rig): crop every section at level 3
    python -m wfield_local.histology_map jobs --cohort PS92-95 > jobs.json
    python -m wfield_local.histology_map.ets jobs.json <crops_dir> 3
    # 2. standardise + Claude's suggestion + painter page (writes the transforms into sections.json)
    python -m wfield_local.histology_map prepare --cohort PS92-95 --painter-html painter.html
    # 3. after painting: copy masks from the painter's database into configs/histology_map/<cohort>/masks/
    # 4. AP levels into sections.json (priya_figures), then fit + figures
    python -m wfield_local.histology_map fit --cohort PS92-95
    python -m wfield_local.histology_map schematic --cohort PS92-95 [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def cohort_dir(c):
    return REPO / "configs" / "histology_map" / c


def load_cohort(c, overrides=None):
    cfg = yaml.safe_load((cohort_dir(c) / "cohort.yaml").read_text())
    cfg.update({k: v for k, v in (overrides or {}).items() if v})
    sp = cohort_dir(c) / "sections.json"
    sections = json.loads(sp.read_text()) if sp.exists() else {}
    return cfg, sections, sp


def _table(cfg):
    import pandas as pd
    p = Path(cfg["section_table"]); p = p if p.is_absolute() else REPO / p
    return pd.read_csv(p)


def cmd_jobs(a):
    cfg, _, _ = load_cohort(a.cohort)
    tab = _table(cfg); jobs = []
    for _, r in tab.iterrows():
        sj = json.loads((Path(cfg["qupath_project"]) / "data" / str(r.entry_id) / "server.json").read_text())
        jobs.append(dict(animal=r.animal, section=int(r.section), image=r.image, slide_section=int(r.slide_section),
                         uri=sj.get("uri") or sj["builder"]["uri"], cx=float((r.ipsi_cx_px + r.contra_cx_px) / 2),
                         cy=float((r.ipsi_cy_px + r.contra_cy_px) / 2)))
    print(json.dumps(jobs, indent=1))


def _standardised(cfg, sections, ann, row):
    from wfield_local.histology_map import sections as S
    sid = f"{row.animal}_s{int(row.section)}"
    img, gi, gc = S.load_raw(Path(cfg["crops_dir"]) / f"{sid}_L3.npz", ann, row)
    rec = sections.get(sid)
    T, base = S.prepare(img, gi, gc, stored=rec if rec and "angle" in rec else None)
    return sid, T, base


def cmd_prepare(a):
    cfg, sections, sp = load_cohort(a.cohort, dict(crops_dir=a.crops, qupath_project=a.qupath))
    from wfield_local import qupath_io
    from wfield_local.histology_map import footprint, painter
    ann = qupath_io.project_annotations(cfg["qupath_project"], keep_geometry=True)
    payloads = []
    for _, row in _table(cfg).iterrows():
        sid, T, base = _standardised(cfg, sections, ann, row)
        sug = footprint.suggest(T)
        d, crop = painter.section_payload(sid, row.animal, row.section, T, sug, sug["mirror"], sug["flags"])
        rec = sections.get(sid, {}); rec.update(base); rec.update(dict(animal=row.animal, section=int(row.section),
                                                                       image=row.image, slide_section=int(row.slide_section)))
        if "crop_x0" not in rec:      # never move the crop of an existing record: painted masks depend on it
            rec.update(crop)
        sections[sid] = rec; payloads.append(d); print(sid, sug["flags"], flush=True)
    sp.write_text(json.dumps(sections, indent=1))
    if a.painter_html:
        painter.build_page(payloads, cfg["cohort"], a.painter_html); print("painter page ->", a.painter_html)


def cmd_fit(a):
    cfg, sections, sp = load_cohort(a.cohort, dict(crops_dir=a.crops, qupath_project=a.qupath, plates_pdf=a.plates))
    from wfield_local import qupath_io
    from wfield_local.histology_map import fit, plates
    ann = qupath_io.project_annotations(cfg["qupath_project"], keep_geometry=True)
    PL = plates.load_plates(cfg["plates_pdf"]); R = {}
    for _, row in _table(cfg).iterrows():
        sid, T, _ = _standardised(cfg, sections, ann, row)
        rec = sections[sid]; figs = rec.get("priya_figures")
        if not figs:
            print(sid, "no AP level yet (priya_figures) -- skipped"); continue
        for f in figs:
            R.setdefault(f, plates.raster(PL[f]))
        res, y_top = fit.fit_section(T, R, figs)
        b = res[0]
        rec.update(figure=b["figure"], bregma=PL[b["figure"]]["bregma"], kx=round(b["kx"], 4), ky=round(b["ky"], 4),
                   ty=round(b["ty"], 4), fit_err_mm=round(b["err_mm"], 4), y_top=y_top,
                   fit_candidates=[dict(figure=r["figure"], err_mm=round(r["err_mm"], 4)) for r in res])
        print(sid, b, flush=True)
    sp.write_text(json.dumps(sections, indent=1))


def cmd_schematic(a):
    cfg, sections, _ = load_cohort(a.cohort, dict(plates_pdf=a.plates))
    from wfield_local import config
    from wfield_local.histology_map import schematic
    if a.out:
        out = Path(a.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("microscope")) / cfg["out_subdir"]
    colors = config.animal_color()
    pdir = cohort_dir(a.cohort) / "atlas_polygons"          # hand-edited atlas shapes win when present
    rows, interp = schematic.render(sections, cohort_dir(a.cohort) / "masks", cfg["plates_pdf"], colors,
                                    out / cfg["out_stem"], cfg["title"], polygons_dir=pdir if pdir.exists() else None)
    for r in rows: print(r)
    print("interpolated:", sorted(interp)); print("->", out / cfg["out_stem"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("jobs", cmd_jobs), ("prepare", cmd_prepare), ("fit", cmd_fit), ("schematic", cmd_schematic)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument("--cohort", required=True)
        p.add_argument("--crops"); p.add_argument("--qupath"); p.add_argument("--plates")
        p.add_argument("--out"); p.add_argument("--painter-html")
    a = ap.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    main()

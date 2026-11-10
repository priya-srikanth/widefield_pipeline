"""Glue-mask BEFORE / AFTER comparison of the finished queue-v20 re-run steps (`*_g`), per animal x epoch first.

WHY. The 11-03 rebuild ran without the painted fiber-glue mask (stat mask 166,974 px instead of 129,770). The fix is on
main 8aed3a7b and queue v20 re-runs every affected step under a `_g` name. Priya (11-09): every readout is a change from
the animal's OWN pre, per animal x epoch first; stars = `boot_delta` CI; any CONCLUSION change goes to her BEFORE page /
deck text. This script lines up each finished `_g` step with a glue-IN counterpart and writes
docs/status/GLUE_BEFORE_AFTER_<date>.md (tables first, the conclusion-change list at the top). Re-run it as more steps
finish; it only READS (M: is never written).

WHERE THE GLUE-IN NUMBERS COME FROM (most glue-in output tables were overwritten in place by the re-run):
  1. the glue-in run's own STEP LOG -- the same command printed its per-animal x epoch tables, bootstrap change-vs-pre
     tables (delta / lo / hi) and map significant-bin counts. Old log = render_daqlk/logs/<step>.log or <step>_r*.log
     (latest exit-0 run started before the re-run, same `# cmd` preferred) or, for the frozen encoder,
     frozen_v7g3_spont/logs/<run>.log (the 11-05 runs whose fits were moved aside);
  2. the frozen encoder's per-area R2: `frozen_encoder_<tag>_roiexact/frozen_musall_r2.csv` (11-06, glue-in, not yet
     re-run). It reproduces the glue-in spont_frA frozen R2 of `session_vs_frozen/gap.csv` (11-09 07:01, glue-in) to
     float rounding on every shared area (checked on every run of this script and printed in the md), so it stands in
     for the overwritten glue-in spont_frA `frozen_musall_r2.csv`;
  3. tables MOVED aside to <root>/_superseded_glue_20261109/<same relative path> (session SD tables).
Anything else: "no glue-in copy".

Usage:  python scripts/glue_before_after.py [--out docs/status/GLUE_BEFORE_AFTER_2026-11-10.md] [--n-boot 4000]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path("C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad")
LOGS = SCRATCH / "render_daqlk" / "logs"
ENC_LOGS = SCRATCH / "frozen_v7g3_spont" / "logs"
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103")
SUPERSEDED = ROOT / "_superseded_glue_20261109"
HV = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801"
SEL = "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70"
E = ROOT / "LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding" / f"{HV}_{SEL}_daqlk"
ITI = ROOT / "LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks"
RERUN_START = dt.datetime(2026, 11, 9, 17, 10)
MANIFESTS = [SCRATCH / "glue_rerun/move_aside_manifest_20261109_171046.csv"] + sorted(SUPERSEDED.glob("move_aside_*.csv"))

EPOCHS = ("pre", "acute", "subacute", "chronic")
POST = ("acute", "subacute", "chronic")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
REL = 0.25          # "large" change
ABS_FLOOR = 0.01    # ... and at least this absolute change (R2 / r / z units), so 0.001 -> 0.002 is not "large"
SIGN_FLOOR = 0.005  # a sign flip counts only when one side is at least this far from 0

# Glue (Priya / A9 audit 11-09): SSs ~50 % glue; VIS / AUD / TEa / RSPagl_L affected; ALM / tjM1 / MO / SSp-m/n/un clean.
EXPECT_CHANGE = ("SSs", "VIS", "RSPagl", "AUD", "TEa", "cortex")
EXPECT_STABLE = ("ALM", "tjM1", "MOs", "MOp", "SSp-m", "SSp-n", "SSp-un")

# frozen-encoder `_g` step -> its 11-05 glue-in run log in frozen_v7g3_spont/logs (same args; checked against the lane
# chain logs below)
ENC_OLD = {"f_frA_g": "frA", "f_frA_pos_g": "frA_pos", "f_frA_basis_g": "frA_basis",
           "f_frA_basis_pixmusall_g": "frA_basis_pixmusall", "f_frAsplit_g": "frAsplit",
           "f_frAsplit_basis_g": "frAsplit_basis", "f_noFR_g": "noFR", "f_noFR_basis_g": "noFR_basis",
           "f_noFR_basis_pixmusall_g": "noFR_basis_pixmusall", "f_frA_ls_g": "frA_ls", "f_frA_ls_basis_g": "frA_ls_basis"}

# per-area frozen R2 (glue-in proxy, glue-out) for the steps that have one
ENC_AREA = {"f_frA_g": (E / "frozen_encoder_gate-attempted_v7g3_t3d_spont_frA_roiexact/frozen_musall_r2.csv",
                        E / "frozen_encoder_gate-attempted_v7g3_t3d_spont_frA/frozen_musall_r2.csv",
                        E / "frozen_encoder_gate-attempted_v7g3_t3d_spont_frA/session_vs_frozen/gap.csv")}
KEY_AREAS = ["cortex", "ALM_left", "ALM_right", "tjM1_left", "tjM1_right", "MOp_left", "MOp_right", "MOs_left",
             "MOs_right", "SSp-m_left", "SSp-m_right", "SSp-n_left", "SSp-n_right", "SSp-un_left", "SSp-un_right",
             "SSs_left", "SSs_right", "VISp_left", "VISp_right", "VISa_left", "VISa_right", "VISam_left", "VISam_right",
             "VISrl_left", "VISrl_right", "RSPagl_left", "RSPagl_right", "AUDp_left", "AUDp_right", "TEa_left",
             "TEa_right"]
UNIQUE = ("target", "spout", "task", "movement", "fr", "fr_dir", "spont", "other")

# moved-aside table pairs (relative to ROOT) per step
SD_TABLES = {
    "e_iti_zscore_pre_g": [f"LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/{HV}/zscore_check/session_sd_roi8_rimlat_almline.csv"],
    "e_iti_split_sd_g": [f"LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/{HV}/zscore_check/session_sd_split_roi8_rimlat_almline.csv"],
    "e_june_ref_g": [f"LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/{HV}_{SEL}_daqlk/june_ref/session_sd_roi8_rimlat_almline.csv",
                     f"LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/{HV}_{SEL}_daqlk/june_ref/session_sd_split_roi8_rimlat_almline.csv"],
}


def short(p) -> str:
    s = str(p).replace("\\", "/")
    return s.replace(f"{HV}_{SEL}", "<HV>_<SEL>").replace(HV, "<HV>")


def expect_tag(text: str) -> str:
    if any(a in text for a in EXPECT_STABLE):
        return "stable-exp"
    if any(a in text for a in EXPECT_CHANGE):
        return "change-exp"
    return ""


def fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:+.{nd}f}" if abs(x) < 1000 else f"{x:.0f}"


# ----------------------------------------------------------------------------------------------- step logs
def log_header(path: Path) -> dict:
    h = {"path": path, "cmd": None, "env": [], "start": None, "exit": None, "code": None}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return h
    for ln in lines:
        if ln.startswith("# cmd:"):
            h["cmd"] = h["cmd"] or ln[6:].strip()
        elif ln.startswith("# env "):
            h["env"].append(ln[6:].strip())
        elif ln.startswith("# start "):
            m = re.match(r"# start (\S+ \S+)\s+code (\S+)", ln)
            if m:
                h["start"] = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                h["code"] = m.group(2)
        elif ln.startswith("# end "):
            m = re.search(r"exit (\d+)", ln)
            h["exit"] = int(m.group(1)) if m else None
    h["env"] = sorted(h["env"])
    return h


def chain_args() -> dict:
    """frozen_v7g3_spont lane chain logs: run name -> (args, start time)."""
    out = {}
    for f in ENC_LOGS.glob("lane*.chain.log"):
        for ln in f.read_text(errors="replace").splitlines():
            m = re.match(r"(\d\d-\d\d \d\d:\d\d:\d\d) (\S+) START .*?args: (.*)$", ln)
            if m:
                out[m.group(2)] = (m.group(3).strip(), m.group(1))
    return out


def old_log_for(step: str, new_h: dict, chains: dict):
    """(header of the glue-in log, note) or (None, reason)."""
    if step in ENC_OLD:
        name = ENC_OLD[step]
        p = ENC_LOGS / f"{name}.log"
        if not p.exists():
            return None, f"no glue-in log {short(p)}"
        h = log_header(p)
        args, when = chains.get(name, (None, None))
        new_args = (new_h["cmd"] or "").split("frozen_encoder", 1)[-1].strip()
        same = args is not None and args.split() == new_args.split()
        h["start_txt"] = f"2026-{when}" if when else "11-05"
        h["code"] = h["code"] or "(not logged; 11-05)"
        return h, ("same args" if same else f"ARGS DIFFER: glue-in `{args}`")
    base = step[:-2]
    cands = [LOGS / f"{base}.log"] + sorted(Path(p) for p in glob.glob(str(LOGS / f"{base}_r*.log")))
    good = []
    for p in cands:
        if not p.exists() or p.name.endswith("_g.log"):
            continue
        h = log_header(p)
        if h["start"] is None or h["start"] >= RERUN_START or h["exit"] != 0:
            continue
        good.append(h)
    if not good:
        return None, "no exit-0 glue-in log of this step (outputs overwritten in place)"
    same = [h for h in good if h["cmd"] == new_h["cmd"] and h["env"] == new_h["env"]]
    pick = max(same or good, key=lambda h: h["start"])
    note = "same cmd + env" if same else "CMD/ENV DIFFER from the re-run"
    pick["start_txt"] = pick["start"].strftime("%Y-%m-%d %H:%M")
    return pick, note


# ----------------------------------------------------------------------------------------------- generic log tables
NUM_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[x%]?$")
RATIO_RE = re.compile(r"^\(?[-+]?[\d.]+/[\d.]+s?\)?,?$")
NOISE_RE = re.compile(
    r"\[joint_locanmf\]|behavior-backup|served from|\[coverage\]|predates|frozen-hit|^\s*wrote |^\s*-> |!!|\[frozen\]"
    r"|template|frozen R\^2 cortex|\btook\b|elapsed|\bETA\b|\d\d:\d\d:\d\d|GB\b|^\s*\[\d+/\d+\]|\{|Traceback|File \""
    r"|\bin \d+(\.\d+)? ?(s|min|h)\b|\d+(\.\d+)? ?s/it|it/s|sessions? per epoch|worker")
BINS_RE = re.compile(r"(\d+) of (\d+) bins")
BINS_LINE_RE = re.compile(r"\.\.\s+(\S+) (\S+) (\S+?):\s+(.*?)(\d+) of (\d+) bins")


def parse_log(path: Path) -> dict:
    """Numeric rows of a step log: key (title, labels, occurrence) -> {labels, nums, names, line, title}."""
    rows, occ = {}, {}
    title, headers, prev = "", [], None
    for raw in path.read_text(errors="replace").splitlines():
        if raw.startswith("#") or not raw.strip() or NOISE_RE.search(raw):
            if not raw.strip():
                prev = None
            continue
        line = raw.rstrip()
        mb = BINS_LINE_RE.search(line)
        if mb:   # map significance count: own key (the line's tail -- edge enrichment, "small" -- comes and goes)
            fig, spout, ep, method, n_sig, n_bins = mb.groups()
            key = ("map significant bins", (fig, spout, ep, re.sub(r"[\d,]+", "#", method).strip(" :")), 1)
            rows[key] = {"labels": list(key[1]), "nums": [float(n_sig), float(n_bins)], "names": None, "groups": None,
                         "title": "map significant bins", "line": line.strip(), "bins": (n_sig, n_bins)}
            continue
        indent = len(line) - len(line.lstrip())
        if line.lstrip().startswith("=") or (indent == 0 and re.search(r"\w=\w", line)):
            # numbers in a title (chance level, trial counts) may themselves move with the mask: key on the words only
            title = re.sub(r"(?<![A-Za-z_\d.])[-+]?\d+(?:\.\d+)?(?![A-Za-z_\d])", "#", line.strip().strip("= ").strip())
            headers, prev = [], None
            continue
        toks = re.sub(r"[\[\],;]", " ", line).split()
        labels, nums, kinds = [], [], []
        for t in toks:
            u = t.strip("():'\"")
            if RATIO_RE.match(t):
                kinds.append("r")
                continue
            if NUM_RE.match(u) or u in ("--", "nan", "NaN"):
                try:
                    nums.append(float(u.rstrip("x%")))
                except ValueError:
                    nums.append(float("nan"))
                kinds.append("n")
            elif u:
                labels.append(u.rstrip(":"))
                kinds.append("l")
        if not nums:
            headers = (headers + [line.split()])[-3:]
            prev = None
            continue
        if prev is not None and indent > 0 and len(labels) < len(prev["labels"]):
            labels = prev["labels"][: len(prev["labels"]) - len(labels)] + labels
        names = None
        ntok = sum(1 for k in kinds if k != "r")
        for h in reversed(headers):
            if len(h) == ntok:
                it = iter(h)
                names = [n for n, k in zip(it, [k for k in kinds if k != "r"]) if k == "n"]
                break
        if names is None:
            for h in reversed(headers):
                ep = [x for x in h if x in EPOCHS]
                if ep and len(ep) == len(nums):
                    names = ep
                    break
        groups = None
        if names and names.count("pre") > 1:
            for h in reversed(headers):
                if len(h) == names.count("pre") and not set(h) & set(EPOCHS):
                    groups = h
                    break
        key0 = (title, tuple(labels))
        occ[key0] = occ.get(key0, 0) + 1
        bins = BINS_RE.search(line)
        rows[key0 + (occ[key0],)] = {"labels": labels, "nums": nums, "names": names, "groups": groups,
                                    "title": title, "line": line.strip(), "bins": bins.groups() if bins else None}
        prev = {"labels": labels}
    return rows


def _flip(a, b, floor=SIGN_FLOOR) -> bool:
    return (math.isfinite(a) and math.isfinite(b) and a * b < 0 and max(abs(a), abs(b)) >= floor)


def _big(a, b) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    d = abs(b - a)
    return d >= ABS_FLOOR and d > REL * max(abs(a), 1e-12)


def dpre(nums, names, groups):
    """[(column label, delta vs own pre)] for an epoch-columned row (each 'pre' column starts a group)."""
    out, pre, g = [], None, -1
    for v, n in zip(nums, names):
        if n == "pre":
            pre, g = v, g + 1
            continue
        if n in POST and pre is not None:
            lab = f"{groups[g]}:{n}" if groups and g < len(groups) else n
            out.append((lab, v - pre))
    return out


def compare_rows(old: dict, new: dict) -> tuple[list, dict]:
    flags, stats = [], {"old": len(old), "new": len(new), "matched": 0, "identical": 0, "shape": 0}
    for k, o in old.items():
        n = new.get(k)
        if n is None:
            continue
        stats["matched"] += 1
        a, b = o["nums"], n["nums"]
        if len(a) != len(b):
            stats["shape"] += 1
            continue
        if all((x == y) or (not math.isfinite(x) and not math.isfinite(y)) for x, y in zip(a, b)):
            stats["identical"] += 1
            continue
        where = " / ".join(x for x in (o["title"][:70], " ".join(o["labels"])) if x)
        tag = expect_tag(where)
        names = o["names"]
        if o["bins"] and n["bins"]:
            no, mo = map(int, o["bins"])
            nn, mn = map(int, n["bins"])
            fo, fn = no / max(mo, 1), nn / max(mn, 1)
            if (no > 0) != (nn > 0):
                flags.append(("star", where, "sig bins", f"{no}/{mo}", f"{nn}/{mn}", tag))
            elif abs(fn - fo) > REL * max(fo, 1e-12) and abs(nn - no) >= 10:
                flags.append(("large", where, "sig-bin fraction", f"{fo:.3f}", f"{fn:.3f}", tag))
            continue
        if names and {"delta", "lo", "hi"} <= set(names):
            i = {x: names.index(x) for x in ("delta", "lo", "hi")}
            so = a[i["lo"]] > 0 or a[i["hi"]] < 0
            sn = b[i["lo"]] > 0 or b[i["hi"]] < 0
            def ci(v, i=i):
                s_ = " *" if (v[i["lo"]] > 0 or v[i["hi"]] < 0) else ""
                return f"{v[i['delta']]:+.3f} [{v[i['lo']]:+.3f},{v[i['hi']]:+.3f}]{s_}"
            if so != sn:
                flags.append(("star", where, "delta CI", ci(a), ci(b), tag))
            elif _flip(a[i["delta"]], b[i["delta"]]):
                flags.append(("sign", where, "delta", ci(a), ci(b), tag))
            elif _big(a[i["delta"]], b[i["delta"]]):
                flags.append(("large", where, "delta", ci(a), ci(b), tag))
            continue
        if "gate=" in o["title"] and len(a) >= 9:          # cd_overlap_null: overlap obs vs matched-null CI
            for nm, (obs, lo, hi) in (("overlap below null", (1, 3, 4)), ("scale above null", (5, 7, 8))):
                st_o = a[obs] < a[lo] if "overlap" in nm else a[obs] > a[hi]
                st_n = b[obs] < b[lo] if "overlap" in nm else b[obs] > b[hi]
                def txt(v, obs=obs, lo=lo, hi=hi):
                    return f"{v[obs]:.3f} vs [{v[lo]:.3f},{v[hi]:.3f}]"
                if st_o != st_n:
                    flags.append(("star", where, nm, txt(a), txt(b), tag))
                elif _big(a[obs], b[obs]):
                    flags.append(("large", where, nm.split()[0], txt(a), txt(b), tag))
            continue
        if names and "pre" in names and set(names) & set(POST) and "delta" not in names:
            for (lab, x), (_, y) in zip(dpre(a, names, o["groups"]), dpre(b, n["names"] or names, o["groups"])):
                if _flip(x, y):
                    flags.append(("sign", where, f"d-pre {lab}", fmt(x), fmt(y), tag))
                elif _big(x, y):
                    flags.append(("large", where, f"d-pre {lab}", fmt(x), fmt(y), tag))
            continue
        for j, (x, y) in enumerate(zip(a, b)):
            col = names[j] if names and j < len(names) else f"col{j + 1}"
            if _flip(x, y):
                flags.append(("sign", where, col, fmt(x), fmt(y), tag))
            elif _big(x, y):
                flags.append(("large", where, col, fmt(x), fmt(y), tag))
    return flags, stats


# ----------------------------------------------------------------------------------------------- encoder per area
def boot_columns(post: dict, pre: dict, n_boot: int):
    from wfield_local.analysis_kit import PCTL, boot_delta_columns   # same draw as boot_delta, column-wise
    r = boot_delta_columns(post, pre, np.random.default_rng(0), n_boot)
    return r, PCTL


def encoder_area(step: str, n_boot: int) -> dict:
    old_p, new_p, gap_p = ENC_AREA[step]
    cols = ["variant", "label", "animal", "epoch", "align", "measure", "area", "r2"]
    old = pd.read_csv(old_p, usecols=cols)
    new = pd.read_csv(new_p, usecols=cols)
    old, new = old[old.measure == "frame"], new[new.measure == "frame"]
    check = None
    if gap_p.exists():
        g = pd.read_csv(gap_p)
        m = g.merge(old.rename(columns={"align": "window"}), on=["label", "window", "area", "variant"])
        check = (len(m), float((m.frozen - m.r2).abs().max()) if len(m) else float("nan"), sorted(m.area.unique()))
    variants = sorted(set(old.variant) & set(new.variant))
    out = {"old": old_p, "new": new_p, "gap": gap_p, "check": check,
           "dropped": sorted(set(old.area) - set(new.area)), "added": sorted(set(new.area) - set(old.area)),
           "variants": variants, "tables": {}}
    for name, df in (("old", old), ("new", new)):
        w = df[df.variant.isin(variants)].pivot_table(index=["label", "animal", "epoch", "align"],
                                                      columns=["variant", "area"], values="r2").sort_index()
        q = {"joint": w["joint"]}
        for u in UNIQUE:
            if f"no_{u}" in variants:
                q[f"uniq_{u}"] = w["joint"] - w[f"no_{u}"]
        out["tables"][name] = q
    res = []
    for qn in out["tables"]["old"]:
        if qn not in out["tables"]["new"]:
            continue
        for align in ("cue", "lick"):
            per = {}
            for name in ("old", "new"):
                t = out["tables"][name][qn].xs(align, level="align")
                areas = [a for a in t.columns]
                per[name] = (t, areas)
            for area in sorted(set(per["old"][1]) | set(per["new"][1])):
                rec = {"q": qn, "align": align, "area": area}
                for name in ("old", "new"):
                    t, areas = per[name]
                    if area not in areas:
                        continue
                    s = t[area].dropna()
                    am = s.groupby(level=["animal", "epoch"]).mean()
                    for a in ANIMALS:
                        for e in POST:
                            if (a, e) in am.index and (a, "pre") in am.index:
                                rec[(name, a, e)] = am[(a, e)] - am[(a, "pre")]
                rec["tag"] = expect_tag(area)
                res.append(rec)
            for name in ("old", "new"):        # pooled boot_delta per epoch, all areas at once
                t, areas = per[name]
                t = t[areas]
                pre = {a: t.xs((a, "pre"), level=["animal", "epoch"]).to_numpy() for a in ANIMALS
                       if (a, "pre") in t.droplevel("label").index}
                for e in POST:
                    post = {a: t.xs((a, e), level=["animal", "epoch"]).to_numpy() for a in ANIMALS
                            if (a, e) in t.droplevel("label").index}
                    r, _ = boot_columns({a: np.nan_to_num(v, nan=np.nanmean(v)) for a, v in post.items()},
                                        {a: np.nan_to_num(v, nan=np.nanmean(v)) for a, v in pre.items()}, n_boot)
                    if r is None:
                        continue
                    pt, lo, hi = r[0], r[1], r[2]
                    for j, area in enumerate(areas):
                        for rec in res:
                            if rec["q"] == qn and rec["align"] == align and rec["area"] == area:
                                rec[(name, "pool", e)] = (pt[j], lo[j], hi[j])
    out["res"] = res
    return out


def star(v) -> bool:
    return v is not None and (v[1] > 0 or v[2] < 0)


def pool_txt(v) -> str:
    return "--" if v is None else f"{v[0]:+.3f}{'*' if star(v) else ''}"


def encoder_flags(enc: dict) -> list:
    flags = []
    for r in enc["res"]:
        where = f"{r['q']} {r['align']} {r['area']}"
        for e in POST:
            po, pn = r.get(("old", "pool", e)), r.get(("new", "pool", e))
            if po is not None and pn is not None:
                if star(po) != star(pn):
                    flags.append(("star", where, f"pooled {e}", pool_txt(po), pool_txt(pn), r["tag"]))
                elif _flip(po[0], pn[0]):
                    flags.append(("sign", where, f"pooled {e}", pool_txt(po), pool_txt(pn), r["tag"]))
            for a in ANIMALS:
                x, y = r.get(("old", a, e)), r.get(("new", a, e))
                if x is None or y is None:
                    continue
                if _flip(x, y):
                    flags.append(("sign", where, f"{a} {e}", fmt(x), fmt(y), r["tag"]))
                elif _big(x, y):
                    flags.append(("large", where, f"{a} {e}", fmt(x), fmt(y), r["tag"]))
    return flags


# ----------------------------------------------------------------------------------------------- moved-aside tables
def sd_tables(step: str) -> list:
    out = []
    for rel in SD_TABLES.get(step, []):
        new_p, old_p = ROOT / rel, SUPERSEDED / rel
        if not (new_p.exists() and old_p.exists()):
            out.append({"new": new_p, "old": old_p, "missing": True})
            continue
        o, n = pd.read_csv(old_p, index_col=0), pd.read_csv(new_p, index_col=0)
        common = sorted(set(o.columns) & set(n.columns), key=list(o.columns).index)
        idx = o.index.intersection(n.index)
        ratio = (n.loc[idx, common] / o.loc[idx, common])
        summ = pd.DataFrame({"median new/old": ratio.median(), "min": ratio.min(), "max": ratio.max(),
                             "n sessions": ratio.notna().sum()})
        out.append({"new": new_p, "old": old_p, "missing": False, "summary": summ,
                    "dropped": sorted(set(o.columns) - set(n.columns)), "added": sorted(set(n.columns) - set(o.columns)),
                    "n_old": len(o), "n_new": len(n)})
    return out


# ----------------------------------------------------------------------------------------------- markdown
PRIORITY = {"star": 0, "sign": 1, "large": 2}
CONCL_FLOOR = 0.01   # a star change on a pooled change smaller than this (both runs) is a CI-edge case, not a conclusion


def _first_num(s: str):
    m = re.search(r"[-+]?\d*\.?\d+", str(s))
    return float(m.group()) if m else float("nan")


def conclusion_level(f) -> bool:
    """True = goes in the top list: a star change with a non-trivial pooled change, or a sign flip of a change that
    is starred in either run. Everything else stays in the step section."""
    kind, _, col, old, new, _ = f
    starred = "*" in str(old) or "*" in str(new)
    if col == "sig bins" or "null" in col:
        return True
    if kind == "star":
        return max(abs(_first_num(old)), abs(_first_num(new))) >= CONCL_FLOOR
    return kind == "sign" and starred


def flag_table(flags: list, cap: int) -> list:
    if not flags:
        return ["_no flagged rows_", ""]
    fl = sorted(flags, key=lambda f: (PRIORITY[f[0]], f[5] != "stable-exp", f[1]))
    lines = ["| flag | row | column | glue-in | glue-out | area expectation |", "|---|---|---|---|---|---|"]
    for f in fl[:cap]:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in f) + " |")
    if len(fl) > cap:
        lines.append(f"| ... | {len(fl) - cap} more flagged rows (re-run the script with --cap) | | | | |")
    return lines + [""]


def counts(flags: list) -> str:
    c = {k: sum(1 for f in flags if f[0] == k) for k in PRIORITY}
    s = {k: sum(1 for f in flags if f[0] == k and f[5] == "stable-exp") for k in PRIORITY}
    return (f"{c['star']} star changes ({s['star']} in expected-stable areas), {c['sign']} sign flips "
            f"({s['sign']} expected-stable), {c['large']} changes > {int(REL * 100)} % ({s['large']} expected-stable)")


def enc_tables(enc: dict) -> list:
    L = []
    for align in ("cue", "lick"):
        L += [f"**Joint frozen R2 ({align} window, frame measure): change vs each animal's own pre, glue-in -> "
              f"glue-out.** Pooled = mean over animals of each animal's change, `boot_delta` CI (* = CI excludes 0).", "",
              "| area | epoch | PS92 | PS93 | PS94 | PS95 | pooled | expectation |", "|---|---|---|---|---|---|---|---|"]
        for area in KEY_AREAS:
            r = next((x for x in enc["res"] if x["q"] == "joint" and x["align"] == align and x["area"] == area), None)
            if r is None:
                continue
            for e in POST:
                cells = []
                for a in ANIMALS:
                    x, y = r.get(("old", a, e)), r.get(("new", a, e))
                    mark = " **(!)**" if (x is not None and y is not None and _flip(x, y)) else ""
                    cells.append(f"{fmt(x)} -> {fmt(y)}{mark}")
                po, pn = r.get(("old", "pool", e)), r.get(("new", "pool", e))
                mark = " **(star!)**" if (po and pn and star(po) != star(pn)) else ""
                L.append(f"| {area} | {e} | " + " | ".join(cells) + f" | {pool_txt(po)} -> {pool_txt(pn)}{mark} | {r['tag']} |")
        L.append("")
    L += ["**Unique contributions (joint - no_<group>), pooled change vs pre, key areas, glue-in -> glue-out** "
          "(per-animal flags are in the flag list below).", "",
          "| quantity | window | area | acute | subacute | chronic |", "|---|---|---|---|---|---|"]
    for r in enc["res"]:
        if r["q"] == "joint" or r["area"] not in ("cortex", "ALM_left", "ALM_right", "MOp_left", "MOp_right",
                                                   "SSp-m_left", "SSp-m_right", "SSs_left", "SSs_right"):
            continue
        cells = []
        for e in POST:
            po, pn = r.get(("old", "pool", e)), r.get(("new", "pool", e))
            mark = " **(star!)**" if (po and pn and star(po) != star(pn)) else ""
            cells.append(f"{pool_txt(po)} -> {pool_txt(pn)}{mark}")
        L.append(f"| {r['q']} | {r['align']} | {r['area']} | " + " | ".join(cells) + " |")
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "docs/status/GLUE_BEFORE_AFTER_2026-11-10.md"))
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--cap", type=int, default=40, help="max flagged rows listed per step")
    a = ap.parse_args()

    chains = chain_args()
    done = sorted(Path(p).name[:-5] for p in glob.glob(str(LOGS / "*_g.done")))
    steps, top = [], []
    for step in done:
        new_log = LOGS / f"{step}.log"
        nh = log_header(new_log)
        oh, note = old_log_for(step, nh, chains)
        rec = {"step": step, "new": nh, "old": oh, "note": note, "flags": [], "stats": None, "enc": None, "sd": []}
        if oh is not None:
            rec["flags"], rec["stats"] = compare_rows(parse_log(oh["path"]), parse_log(new_log))
        if step in ENC_AREA:
            proxy = ENC_AREA[step][0]
            if proxy.exists() and dt.datetime.fromtimestamp(proxy.stat().st_mtime) > RERUN_START:
                # 11-10: the `_roiexact` re-run rewrote the glue-in proxy in place at 11:16, so it is glue-OUT now and
                # the comparison would be glue-out vs glue-out (all star changes vanish). Refuse it; the 11-10 01:28
                # report (commit a52a1afd) holds the valid per-area tables, made before the overwrite.
                note += (f"; per-area R2 SKIPPED: glue-in proxy {proxy.name} rewritten by the re-run "
                         f"({dt.datetime.fromtimestamp(proxy.stat().st_mtime):%m-%d %H:%M}); valid per-area tables = "
                         "the 11-10 01:28 report (commit a52a1afd)")
            else:
                rec["enc"] = encoder_area(step, a.n_boot)
                rec["flags"] += encoder_flags(rec["enc"])
        rec["sd"] = sd_tables(step)
        if oh is None and rec["enc"] is None and not any(not s.get("missing") for s in rec["sd"]):
            rec["status"] = "no glue-in copy"
        elif (rec["stats"] is not None and rec["enc"] is None and not rec["sd"]
              and (rec["stats"]["matched"] == 0 or (rec["stats"]["matched"] < 5 and not rec["flags"]))):
            rec["status"] = "no glue-in copy (neither log prints readouts; outputs overwritten in place)"
        else:
            rec["status"] = "compared"
        for f in rec["flags"]:
            if f[0] in ("star",) or (f[0] == "sign" and ("pooled" in f[2] or "delta" in f[2])):
                top.append((step,) + f + (conclusion_level(f),))
        steps.append(rec)

    L = [f"# Glue mask before / after -- per animal x epoch (generated {dt.datetime.now():%Y-%m-%d %H:%M})", "",
         "Generated by `scripts/glue_before_after.py` (read-only on M:). Re-run it as more `_g` steps finish; it "
         "rewrites this file. Glue-in = the 11-03 rebuild WITHOUT the painted fiber-glue mask; glue-out = queue v20 "
         "re-run on main 8aed3a7b. Every readout below is a change from the animal's OWN pre first; pooled = mean "
         "over animals; stars = `boot_delta` CI (95 %, paired animals -> sessions). Thresholds: large = |change| > "
         f"{int(REL * 100)} % and >= {ABS_FLOOR}; sign flip counted when one side is >= {SIGN_FLOOR} from 0.", "",
         "**Expected to change:** SSs (~50 % glue), VIS, AUD, TEa, RSPagl_L, whole-cortex means; glue-only areas "
         "dropped. **Expected stable:** ALM, tjM1, MOs / MOp, SSp-m / SSp-n / SSp-un (and LocaNMF decoders).", "",
         "**Caveat:** the glue-in runs used older code (11-05 .. 11-08 commits, listed per step); a difference can "
         "also come from code / input changes between the two runs, not only from the mask.", "",
         "## Conclusion changes to bring to Priya (BEFORE page / deck text)", ""]
    hdr = ["| step | flag | row | column | glue-in | glue-out | area expectation |", "|---|---|---|---|---|---|---|"]
    srt = lambda rows: sorted(rows, key=lambda t: (PRIORITY[t[1]], t[6] != "stable-exp", t[0], t[2]))
    main_rows = [t[:7] for t in top if t[7]]
    edge_rows = [t[:7] for t in top if not t[7]]
    L += ["**A. Candidate conclusion changes** -- a star change (CI crossing 0, significant bins / matched-null "
          f"status appearing or disappearing) on a pooled change >= {CONCL_FLOOR} in either run, or a sign flip of a "
          "change that is starred in either run. Expected-stable rows first.", ""]
    L += (hdr + ["| " + " | ".join(str(x).replace("|", "/") for x in t) + " |" for t in srt(main_rows)] + [""]
          if main_rows else ["_none so far_", ""])
    L += [f"**B. Borderline / not conclusion-level ({len(edge_rows)})** -- star changes on pooled changes < "
          f"{CONCL_FLOOR} (CI-edge cases with near-identical point estimates) and sign flips of changes that are "
          "non-significant in both runs.", ""]
    L += (hdr + ["| " + " | ".join(str(x).replace("|", "/") for x in t) + " |" for t in srt(edge_rows)] + [""]
          if edge_rows else ["_none_", ""])

    L += ["## Coverage (finished `_g` steps)", "",
          "| step | status | glue-in source | glue-in run | rows matched / identical | flags |", "|---|---|---|---|---|---|"]
    for r in steps:
        oh = r["old"]
        src = short(oh["path"].relative_to(SCRATCH)) if oh else "--"
        run = f"{oh.get('start_txt', '')} code {oh.get('code')} ({r['note']})" if oh else r["note"]
        extra = []
        if r["enc"]:
            extra.append("+ per-area frozen R2 (roiexact proxy)")
        if any(not s.get("missing") for s in r["sd"]):
            extra.append("+ moved-aside SD tables")
        st = r["stats"]
        m = f"{st['matched']} / {st['identical']}" if st else "--"
        L.append(f"| {r['step']} | {r['status']} | {src} {' '.join(extra)} | {run} | {m} | {counts(r['flags'])} |")
    L.append("")

    for r in steps:
        L += [f"## {r['step']}", ""]
        nh, oh = r["new"], r["old"]
        L += [f"- glue-out log: `{short(nh['path'])}` (start {nh['start']}, code {nh['code']})",
              f"- glue-out cmd: `{short(nh['cmd'])}`"]
        if oh:
            L += [f"- glue-in log: `{short(oh['path'])}` ({oh.get('start_txt')}, code {oh.get('code')}; {r['note']})"]
        else:
            L += [f"- glue-in: {r['note']}"]
        L.append(f"- status: **{r['status']}**; {counts(r['flags'])}")
        if r["stats"]:
            s = r["stats"]
            L.append(f"- log rows: glue-in {s['old']}, glue-out {s['new']}, matched {s['matched']}, identical "
                     f"{s['identical']}, shape changed {s['shape']}")
        L.append("")
        enc = r["enc"]
        if enc:
            ck = enc["check"]
            L += [f"Per-area frozen R2: glue-in proxy `{short(enc['old'])}` vs glue-out `{short(enc['new'])}`.",
                  f"Proxy check vs glue-in `{short(enc['gap'])}`: {ck[0]} shared rows (areas {', '.join(ck[2])}), "
                  f"max |diff| {ck[1]:.1e}." if ck else "Proxy check: gap.csv missing.",
                  f"Areas in glue-in only (dropped / not in this glue-out run): {', '.join(enc['dropped']) or 'none'}. "
                  f"New in glue-out: {', '.join(enc['added']) or 'none'}. ALM / tjM1 are ROI targets: their glue-out "
                  "numbers come with the `_roiexact` / pixmusall re-runs (not finished).", ""]
            L += enc_tables(enc)
        for sd in r["sd"]:
            if sd.get("missing"):
                L += [f"Moved-aside table pair missing: `{short(sd['old'])}` / `{short(sd['new'])}`", ""]
                continue
            L += [f"Session SD table (area-trace scale used for z-scoring): glue-in `{short(sd['old'])}` "
                  f"({sd['n_old']} sessions) vs glue-out `{short(sd['new'])}` ({sd['n_new']} sessions). Dropped "
                  f"columns: {', '.join(sd['dropped']) or 'none'}; added: {', '.join(sd['added']) or 'none'}.", "",
                  "| column | median new/old | min | max | n | expectation |", "|---|---|---|---|---|---|"]
            for c, row in sd["summary"].iterrows():
                L.append(f"| {c} | {row['median new/old']:.3f} | {row['min']:.3f} | {row['max']:.3f} | "
                         f"{int(row['n sessions'])} | {expect_tag(c)} |")
            L.append("")
        L += [f"Flagged rows ({counts(r['flags'])}; star / sign first, expected-stable first):", ""]
        L += flag_table(r["flags"], a.cap)

    L += ["## Inputs", "", "Move-aside manifests: " + ", ".join(f"`{short(p)}`" for p in MANIFESTS), ""]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {a.out}  ({len(steps)} steps, {len(top)} conclusion-level flags)")


if __name__ == "__main__":
    main()

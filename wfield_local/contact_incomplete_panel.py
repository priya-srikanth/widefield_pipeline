"""CONTACT vs INCOMPLETE: the frozen spout decoder's accuracy by camera class, per spout (Priya 11-04; sections D / G).

One standard panel for both sections. Per alignment (cue, precue) one row; per epoch (pre, acute, subacute, chronic)
one column; x = the six spouts in canonical order (close ipsi, close mid, close contra, far ipsi, far mid, far contra;
`orofacial_cohort.POS_ORDER`), CONTACT and INCOMPLETE side by side at each spout. Each point is ONE SESSION (open
marker: shape = animal, colour = epoch); no line joins epochs. Chance 1/6 (dashed).

DATA (`rows_from_*`, all to one long table of per-session counts n / k-correct per class x spout):
  * POST sessions: section G's by-class panels (`poststroke_compare._by_class_panels`, the frozen decoder's full model on
    the post session's camera-class rows: ``section_g.json`` arms -> confusion -> by_class -> post).
  * PRE sessions (and any session section D scored): section D's scored WORKING rows split by `working_trials.g_group`
    (`locanmf_frozen_decoder.pooled_frozen_loso` ``confusion_by_class``; pre sessions by their LOSO model).
  The decoder's scored frame (its centre rows: working | contact, `spout_scaling.score_centre_rows`) is the SAME for
  both classes and is whatever the producing run used; the panel prints the tag it was given (``centre``), never
  assumes one.

PRE REFERENCE per class and spout: the pre sessions' pooled accuracy for THAT class when its pooled pre cell has
>= MIN_N rows; otherwise pre WORKING (contact + incomplete), drawn dotted and labelled "pre working". Never post
incomplete against pre contact.

THIN CELLS: a session x class x spout cell with fewer than MIN_N (10) rows is OMITTED from the dots and the test, and
counted: the number of omitted sessions is printed under the spout ("n<10: k"). Stars: in the top band
(`sig_marks.star_top` + `reserve_headroom`), when the paired animals -> sessions bootstrap (`analysis_kit.boot_delta`)
of post - pre (per class, per spout, per epoch; the pre reference above) excludes 0; no top spine.

figdata + ``--plot-only``: `main` saves the long table and redraws from it alone.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_N = 10
CHANCE = 1 / 6
CLASSES = ("contact", "incomplete")
EPOCHS = ("pre", "acute", "subacute", "chronic")
ALIGNS = ("cue", "precue")
CLASS_OFFSET = {"contact": -0.17, "incomplete": 0.17}
#: DISPLAY_ORDER position names -> lesion-relative spout names (as `attempt_coding.LESION_NAME`)
LESION_NAME = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
               "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}


def _pos_order():
    try:
        from wfield_local.orofacial_cohort import POS_ORDER
        return tuple(POS_ORDER)
    except Exception:                                                    # noqa: BLE001
        return ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")


def _names():
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES
    return [LESION_NAME.get(POSITION_NAMES[c], POSITION_NAMES[c]) for c in DISPLAY_ORDER]


def _diag_rows(M, n, meta):
    """Rows from one row-normalised confusion (DISPLAY_ORDER) and its per-true-position n: k = diag * n."""
    out = []
    for j, sp in enumerate(_names()):
        nj = int(n[j]) if n is not None else 0
        if not nj:
            continue
        acc = M[j][j]
        out.append(dict(meta, spout=sp, n=nj, k=float(acc) * nj if acc is not None and np.isfinite(acc) else np.nan))
    return out


def rows_from_by_class(by_class, *, animal, session, epoch, align, centre, source="G"):
    """Long rows from one section G ``by_class`` record (`_by_class_panels`): its POST block, contact + incomplete."""
    rows = []
    for cls in CLASSES:
        p = (by_class.get("post") or {}).get(cls) or {}
        if p.get("matrix") is None:
            continue
        rows += _diag_rows(p["matrix"], p.get("n_per_true_position"),
                           dict(animal=animal, session=session, epoch=epoch, align=align, cls=cls, centre=centre,
                                source=source))
    return rows


def rows_from_section_d(res, *, animal, align, epoch_of):
    """Long rows from a `pooled_frozen_loso` result's ``confusion_by_class`` (COUNT matrices, DISPLAY_ORDER rows)."""
    rows = []
    for lab, per in (res.get("confusion_by_class") or {}).items():
        for cls in CLASSES:
            C = per.get(cls)
            if C is None:
                continue
            C = np.asarray(C, float)
            for j, sp in enumerate(_names()):
                nj = int(C[j].sum())
                if nj:
                    rows.append(dict(animal=animal, session=lab, epoch=epoch_of(lab), align=align, cls=cls,
                                     centre=res.get("scored_centre"), source="D", spout=sp, n=nj, k=float(C[j, j])))
    return rows


def rows_from_section_g(rec_by_session, *, epoch_of, centre, arm="all", conds=(("post-cue", "cue"),
                                                                               ("pre-cue", "precue"))):
    """Long rows from a whole ``section_g.json`` record: every post session's by-class panel per alignment."""
    rows = []
    for lab, rec in (rec_by_session or {}).items():
        if rec.get("phase_tag") != "post":
            continue
        conf = ((rec.get("arms") or {}).get(arm) or {}).get("confusion") or {}
        for cond, align in conds:
            bc = (conf.get(cond) or {}).get("by_class")
            if bc:
                rows += rows_from_by_class(bc, animal=rec.get("animal"), session=lab, epoch=epoch_of(lab),
                                           align=align, centre=centre)
    return rows


def panel_table(rows, min_n=MIN_N):
    """The panel's long table: one row per session x class x spout with ``acc`` = k / n and ``thin`` = n < min_n."""
    T = pd.DataFrame(rows)
    if T.empty:
        return T
    T = T.groupby(["animal", "session", "epoch", "align", "cls", "spout"], as_index=False).agg(
        n=("n", "sum"), k=("k", "sum"), centre=("centre", "first"))
    T["acc"] = T.k / T.n
    T["thin"] = T.n < min_n
    return T


def pre_reference(T, min_n=MIN_N):
    """{(align, cls, spout): (value, "same" | "working")}: pooled pre accuracy of the class, else of pre working."""
    out = {}
    pre = T[T.epoch == "pre"]
    for (al, sp), g in pre.groupby(["align", "spout"]):
        w = g[g.cls.isin(CLASSES)]
        wv = (w.k.sum() / w.n.sum()) if w.n.sum() else np.nan
        for cls in CLASSES:
            c = g[g.cls == cls]
            if c.n.sum() >= min_n:
                out[(al, cls, sp)] = (float(c.k.sum() / c.n.sum()), "same")
            else:
                out[(al, cls, sp)] = (float(wv), "working")
    return out


def _stars(T, ref, al, ep, cls, sp, rng):
    """'*' when the paired bootstrap of post - pre excludes 0 (animals with both, non-thin cells), else ''."""
    from wfield_local import analysis_kit as ak
    post = T[(T["align"] == al) & (T.epoch == ep) & (T.cls == cls) & (T.spout == sp) & ~T.thin]
    refk = "same" if ref.get((al, cls, sp), (None, "working"))[1] == "same" else "working"
    pcls = [cls] if refk == "same" else list(CLASSES)
    pre = T[(T["align"] == al) & (T.epoch == "pre") & T.cls.isin(pcls) & (T.spout == sp) & ~T.thin]
    if refk == "working" and len(pre):
        pre = pre.groupby(["animal", "session"], as_index=False).agg(n=("n", "sum"), k=("k", "sum"))
        pre["acc"] = pre.k / pre.n
    a = {an: list(g.acc) for an, g in post.groupby("animal")}
    b = {an: list(g.acc) for an, g in pre.groupby("animal")}
    if len(set(a) & set(b)) < 3:
        return ""
    iv = ak.boot_delta(a, b, rng)
    if iv is None:
        return ""
    lo, hi = (iv.lo, iv.hi) if hasattr(iv, "lo") else (iv[1], iv[2])
    return "*" if (lo > 0 or hi < 0) else ""


def draw(T, out_png, *, min_n=MIN_N, title=None, seed=0):
    """The panel from the long table alone (``--plot-only`` path). Returns the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import sig_marks
    try:
        from wfield_local.orofacial_cohort import ANIMAL_MARKER
    except Exception:                                                    # noqa: BLE001
        ANIMAL_MARKER = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    try:
        from wfield_local.cd_trajectories import EPOCH_COLOR
    except Exception:                                                    # noqa: BLE001
        EPOCH_COLOR = {"pre": "0.2", "acute": "tab:red", "subacute": "tab:orange", "chronic": "tab:blue"}
    order = _pos_order()
    ref = pre_reference(T, min_n)
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(len(ALIGNS), len(EPOCHS), figsize=(3.4 * len(EPOCHS), 3.0 * len(ALIGNS)), sharey=True,
                            squeeze=False)
    centre = ", ".join(sorted({str(c) for c in T.get("centre", pd.Series(dtype=object)).dropna().unique()})) or "?"
    for r, al in enumerate(ALIGNS):
        for c, ep in enumerate(EPOCHS):
            ax = axs[r][c]
            ax.axhline(CHANCE, color="0.5", ls="--", lw=0.8, zorder=0)
            for j, sp in enumerate(order):
                for cls in CLASSES:
                    x = j + CLASS_OFFSET[cls]
                    cell = T[(T["align"] == al) & (T.epoch == ep) & (T.cls == cls) & (T.spout == sp)]
                    ok, thin = cell[~cell.thin], cell[cell.thin]
                    for _, q in ok.iterrows():
                        ax.scatter(x + rng.uniform(-.05, .05), q.acc, s=16, facecolors="none",
                                   edgecolors=EPOCH_COLOR.get(ep, "k"), marker=ANIMAL_MARKER.get(q.animal, "o"),
                                   linewidths=0.9, zorder=3)
                    if len(thin):
                        ax.text(x, 0.01, f"n<{min_n}:{len(thin)}", rotation=90, fontsize=5, ha="center",
                                va="bottom", color="0.4", transform=ax.get_xaxis_transform())
                    if ep != "pre" and (al, cls, sp) in ref:
                        v, kind = ref[(al, cls, sp)]
                        if np.isfinite(v):
                            ax.hlines(v, x - .13, x + .13, color="k", lw=1.2, ls="-" if kind == "same" else ":",
                                      zorder=4)
                        s = _stars(T, ref, al, ep, cls, sp, rng)
                        if s:
                            sig_marks.star_top(ax, x, s, color="k", row=0)
            ax.set_xticks(range(len(order)), [o.replace("_", "\n") for o in order], fontsize=6)
            ax.set_title(f"{al} | {ep}", fontsize=8)
            if c == 0:
                ax.set_ylabel("frozen decoder accuracy", fontsize=8)
            ax.set_ylim(0, 1.0)
    h = [Line2D([], [], marker=m, ls="", mfc="none", mec="k", label=a) for a, m in ANIMAL_MARKER.items()]
    h += [Line2D([], [], color="k", lw=1.2, label="pre, same class"),
          Line2D([], [], color="k", lw=1.2, ls=":", label="pre working (class absent pre)"),
          Line2D([], [], color="0.5", ls="--", label="chance 1/6")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title or f"CONTACT (left) vs INCOMPLETE (right) per spout; scored-frame centre: {centre}; cells with "
                 f"n < {min_n} omitted", fontsize=8, y=1.06)
    sig_marks.reserve_headroom(fig)
    try:
        from wfield_local.orofacial_cohort import _tidy
        _tidy(fig)
    except Exception:                                                    # noqa: BLE001
        pass
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section-g", type=Path, help="section_g.json (post sessions' by-class panels)")
    ap.add_argument("--section-d", type=Path, nargs="*", default=[],
                    help="pickled / JSON pooled_frozen_loso results: {(animal, align): res}")
    ap.add_argument("--out", type=Path, required=True, help="output directory (figures/ + figdata/)")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    fd = a.out / "figdata" / "contact_incomplete_panel.pkl"
    if a.plot_only:
        if not fd.exists():
            raise SystemExit(f"--plot-only: no figure data at {fd}")
        T = pickle.loads(fd.read_bytes())
    else:
        import json

        from wfield_local import epochs, spout_scaling
        rows = []
        if a.section_g:
            rows += rows_from_section_g(json.loads(a.section_g.read_text()), epoch_of=epochs.epoch_of,
                                        centre=spout_scaling.score_centre_tag())
        for p in a.section_d:
            D = pickle.loads(p.read_bytes()) if p.suffix == ".pkl" else json.loads(p.read_text())
            for key, res in D.items():
                an, al = key if isinstance(key, tuple) else key.split("|")
                rows += rows_from_section_d(res, animal=an, align=al, epoch_of=epochs.epoch_of)
        T = panel_table(rows)
        fd.parent.mkdir(parents=True, exist_ok=True)
        fd.write_bytes(pickle.dumps(T))
    draw(T, a.out / "figures" / "contact_incomplete_panel.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

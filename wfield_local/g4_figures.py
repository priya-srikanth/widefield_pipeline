"""SECTION G4 figures with the CAMERA-CLASS discriminator as the PRIMARY readout (coordinator 11-04, follow-on 3).

Primary panel per alignment (cue, precue): `poststroke_compare.g4_camclass` -- the fraction of each camera group's
rows a pre-stroke WORKING-vs-NO_RESPONSE discriminator reads as working (Priya 11-09, as G1-G3: negatives = the raw
camera class no_response within the working period; unknown never trained), spout-balanced over spouts with >= 10 rows,
one OPEN dot per post session (shape = animal, color = epoch), groups on x (contact, incomplete, jaw only, no response,
unknown (jaw unjudged), quit; the rebuild's stopped / stopped-licking groups when present; records from before 11-09
carry only the pooled "none" group, drawn as such), epochs side by side within a group, no lines across
epochs. Pre reference per group: the held-out pre value of the SAME group (solid tick), or pre WORKING when the group
has no pre cell (dotted tick, labeled). Stars in the top band (`sig_marks.star_top` + `reserve_headroom`) when the
paired animals -> sessions bootstrap (`analysis_kit.boot_delta`) of post - pre excludes 0; no top spine.

Robustness panel (labeled): the DAQ lick-vs-no-lick discriminator (`fits_engaged_distribution`): post no-lick value
per session against the pre engaged / no-lick reference means.

Data come from ``section_g.json`` only; ``--plot-only`` redraws from the saved figure data (figdata/g4_camclass.pkl).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALIGNS = ("cue", "precue")
EPOCHS = ("acute", "subacute", "chronic")
EP_OFF = {"acute": -0.22, "subacute": 0.0, "chronic": 0.22}


def g4_table(rec, epoch_of):
    """``(camera rows, daq rows, pre rows)`` long tables from a section G record {label: session record}."""
    cam, daq, pre = [], [], {}
    for lab, r in sorted((rec or {}).items()):
        if r.get("phase_tag") != "post":
            continue
        an, ep = r.get("animal"), epoch_of(lab)
        for al in ALIGNS:
            g = r.get(f"g4_camclass_{al}")
            if g and "post" in g:
                for grp, c in g["post"].items():
                    cam.append(dict(animal=an, session=lab, epoch=ep, align=al, group=grp,
                                    value=c.get("frac_spout_balanced"), n=c.get("n"),
                                    pre_reference=c.get("pre_reference"), pre_reference_is=c.get("pre_reference_is"),
                                    frame=g.get("frame")))
                for grp, c in (g.get("pre_reference") or {}).items():
                    pre.setdefault((an, al, grp), [v for v in (c.get("per_session") or []) if v is not None])
            f = r.get(f"fits_engaged_{al}")
            if f and "post_value" in f:
                daq.append(dict(animal=an, session=lab, epoch=ep, align=al, post_nolick=f["post_value"],
                                train=(f.get("train_classes") or {}).get("groups", "daq"),
                                ref_engaged=(f.get("engaged") or {}).get("mean"),
                                ref_nolick=(f.get("nolick") or {}).get("mean")))
    P = pd.DataFrame([dict(animal=a, align=al, group=g, value=v) for (a, al, g), vs in pre.items() for v in vs])
    return pd.DataFrame(cam), pd.DataFrame(daq), P


def g4_order(C):
    """The x order of the groups: `working_trials.G_ALL` with "none" replaced by its raw split (no_response, unknown;
    Priya 11-09) when the records carry it, else the pooled "none" (records from before 11-09)."""
    from wfield_local import working_trials as wt
    split = len(C) and C.group.isin(["no_response", "unknown"]).any()
    out = []
    for g in wt.G_ALL:
        out += ["no_response", "unknown"] if (g == "none" and split) else [g]
    return out


def _star(C, P, al, grp, ep, rng):
    from wfield_local import analysis_kit as ak
    post = C[(C["align"] == al) & (C.group == grp) & (C.epoch == ep)].dropna(subset=["value"])
    pre = P[(P["align"] == al) & (P.group == grp)] if len(P) else P
    a = {an: list(g.value) for an, g in post.groupby("animal")}
    b = {an: list(g.value) for an, g in pre.groupby("animal")} if len(pre) else {}
    if len(set(a) & set(b)) < 3:
        return ""
    iv = ak.boot_delta(a, b, rng)
    return "*" if iv is not None and (iv.lo > 0 or iv.hi < 0) else ""


def draw(C, D, P, out_png, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import poststroke_compare as pc
    from wfield_local import sig_marks
    from wfield_local import working_trials as wt
    try:
        from wfield_local.orofacial_cohort import ANIMAL_MARKER
    except Exception:                                                    # noqa: BLE001
        ANIMAL_MARKER = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    try:
        from wfield_local.cd_trajectories import EPOCH_COLOR
    except Exception:                                                    # noqa: BLE001
        EPOCH_COLOR = {"acute": "tab:red", "subacute": "tab:orange", "chronic": "tab:blue"}
    rng = np.random.default_rng(seed)
    groups = [g for g in g4_order(C) if len(C) and (C.group == g).any()] or list(wt.G_GROUPS)
    labels = {**wt.G_LABELS, **pc.G4_SPLIT_LABELS}
    fig, axs = plt.subplots(len(ALIGNS), 2, figsize=(11, 3.2 * len(ALIGNS)), squeeze=False,
                            gridspec_kw={"width_ratios": [3, 1.2]})
    for r, al in enumerate(ALIGNS):
        ax = axs[r][0]
        for j, grp in enumerate(groups):
            for ep in EPOCHS:
                x = j + EP_OFF[ep]
                cell = C[(C["align"] == al) & (C.group == grp) & (C.epoch == ep)].dropna(subset=["value"]) \
                    if len(C) else C
                for _, q in cell.iterrows():
                    ax.scatter(x + rng.uniform(-.04, .04), q.value, s=16, facecolors="none",
                               edgecolors=EPOCH_COLOR.get(ep, "k"), marker=ANIMAL_MARKER.get(q.animal, "o"), lw=.9)
                s = _star(C, P, al, grp, ep, rng) if len(C) else ""
                if s:
                    sig_marks.star_top(ax, x, s, color=EPOCH_COLOR.get(ep, "k"))
            ref = C[(C["align"] == al) & (C.group == grp)] if len(C) else C
            if len(ref) and ref.pre_reference.notna().any():
                kind = ref.pre_reference_is.dropna().iloc[0] if ref.pre_reference_is.notna().any() else grp
                ax.hlines(float(np.nanmean(ref.pre_reference.astype(float))), j - .35, j + .35, color="k",
                          lw=1.2, ls="-" if kind == grp else ":")
        ax.axhline(0.5, color="0.6", ls="--", lw=.7)
        ax.set_xticks(range(len(groups)), [labels.get(g, g) for g in groups], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction read as WORKING", fontsize=8)
        fr = C.frame.dropna().iloc[0] if len(C) and C.frame.notna().any() else "camclass"
        ax.set_title(f"{al}: G4 camera-class discriminator, pre working vs no response (PRIMARY; {fr})", fontsize=8)
        bx = axs[r][1]
        dd = D[D["align"] == al] if len(D) else D
        for k, ep in enumerate(EPOCHS):
            cell = dd[dd.epoch == ep] if len(dd) else dd
            for _, q in cell.iterrows():
                bx.scatter(k + rng.uniform(-.06, .06), q.post_nolick, s=16, facecolors="none",
                           edgecolors=EPOCH_COLOR.get(ep, "k"), marker=ANIMAL_MARKER.get(q.animal, "o"), lw=.9)
        if len(dd):
            bx.axhline(float(np.nanmean(dd.ref_engaged.astype(float))), color="k", lw=1)
            bx.axhline(float(np.nanmean(dd.ref_nolick.astype(float))), color="k", lw=1, ls=":")
        bx.set_xticks(range(len(EPOCHS)), EPOCHS, fontsize=7)
        bx.set_ylim(0, 1)
        # the G1-G3 discriminator trains camera classes under the camera grouping since 11-08 ((f) 1)
        cam = len(dd) and "train" in dd and (dd.train == "camera").any()
        bx.set_title(("SESSION-z FRAME: working vs no response" if cam else "ROBUSTNESS: DAQ lick vs no-lick")
                     + "\n(post no-lick; pre engaged — / no-lick ···)", fontsize=7)
    h = [Line2D([], [], marker=m, ls="", mfc="none", mec="k", label=a) for a, m in ANIMAL_MARKER.items()]
    h += [Line2D([], [], color="k", lw=1.2, label="pre, same group"),
          Line2D([], [], color="k", lw=1.2, ls=":", label="pre working (group absent pre)")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), fontsize=6, frameon=False, bbox_to_anchor=(.5, 1.03))
    sig_marks.reserve_headroom(fig)
    try:
        from wfield_local.orofacial_cohort import _tidy
        _tidy(fig)
    except Exception:                                                    # noqa: BLE001
        pass
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        from wfield_local import trial_set_text as tst
        try:
            text = trial_set_sentence(C, D, groups=groups)
        except Exception as ex:                                        # noqa: BLE001 - a caption never kills a figure
            print(f"  [trial set] no caption ({type(ex).__name__}: {ex})", flush=True)
            text = None
        with tst.active(None, text=text):
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


def trial_set_sentence(C=None, D=None, groups=None) -> str:
    """The G4 figure's trial set (Priya 11-09): the camera-class discriminator's training / scored rows (left) AND the
    right panel's discriminator (camera WORKING vs NO RESPONSE since 11-08, else DAQ lick vs no-lick), joined with
    `trial_set_text.union`. ``groups`` = the camera groups drawn (default `working_trials.G_ALL` + none's raw split);
    the z-frame center from the records' ``frame`` (else the run's `poststroke_compare.g4_centre`)."""
    from wfield_local import poststroke_compare as pc
    from wfield_local import section_g_figures as sgf
    from wfield_local import trial_set_text as tst
    from wfield_local import working_trials as wt
    labels = {**wt.G_LABELS, **pc.G4_SPLIT_LABELS}
    if not groups:                       # the 11-09 order: "none" drawn as its raw split (`g4_order`)
        groups = [x for g in wt.G_ALL for x in (pc.G13_NONE_SPLIT if g == "none" else (g,))]
    fr = (C.frame.dropna().iloc[0] if C is not None and len(C) and "frame" in C and C.frame.notna().any()
          else pc.g4_frame_tag())
    centre = str(fr).split("/z-", 1)[1].split("/", 1)[0] if "/z-" in str(fr) else pc.g4_centre()
    scored_only = [labels.get(g, g) for g in groups if g not in pc.G4_POS + pc.G4_NEG + ("none",)]
    left = (f"G4 (left): every trial of the post session at its preserved spouts, scored per camera group "
            f"({' / '.join(labels.get(g, g) for g in groups)}); discriminator trained on pre-stroke "
            f"{sgf.working_rows()} vs camera NO RESPONSE trials (raw class no_response: no lick seen on camera, "
            f"within the working period), spout-balanced, each session z-scored on its camera {centre} rows; "
            f"never trained on: {' / '.join(scored_only)}")
    if D is None or not len(D):
        return f"trials: {left}"
    cam = "train" in D and bool((D.train == "camera").any())
    right = "right panel: " + sgf._core(sgf.g13_sentence([{"train_classes": {"groups": "camera" if cam else "daq"}}]))
    return tst.union(left, right, unit="trials")


def render(rec, out, prefix="section_g", figdata=None):
    """Build the table from ``rec`` (section_g.json), save the figure data, draw. Returns the PNG path or None."""
    from wfield_local import epochs
    C, D, P = g4_table(rec, epochs.epoch_of)
    if C.empty:
        return None
    fd = Path(figdata or Path(out) / "figdata") / f"{prefix}_g4_camclass.pkl"
    fd.parent.mkdir(parents=True, exist_ok=True)
    fd.write_bytes(pickle.dumps((C, D, P)))
    png = Path(out) / f"{prefix}_g4_camclass.png"
    draw(C, D, P, png)
    return png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section-g", type=Path, help="section_g.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="section_g")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    if a.plot_only:
        fd = a.out / "figdata" / f"{a.prefix}_g4_camclass.pkl"
        if not fd.exists():
            raise SystemExit(f"--plot-only: no figure data at {fd}")
        C, D, P = pickle.loads(fd.read_bytes())
        draw(C, D, P, a.out / f"{a.prefix}_g4_camclass.png")
        return 0
    return 0 if render(json.loads(a.section_g.read_text()), a.out, a.prefix) else 1


if __name__ == "__main__":
    sys.exit(main())

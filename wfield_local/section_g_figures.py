"""Render EVERY Section G figure from `section_g.json`, one panel per POST-STROKE SESSION.

Reads the JSON rather than recomputing, so a figure and the number quoted beside it cannot drift
apart: if they disagree, one of the two files is stale and that is visible rather than silent.

This and `poststroke_section_g` replace four scratchpad scripts. One opened with
`for an in ("PS94", "PS95")` and another with `if config.session_phase(a, "0817") != "post"`, so PS92
and PS93 were absent from every figure downstream and PS93 had no behaviour slide at all. Twelve of
the deck's figures were still drawn from files those scripts wrote on 2026-08-18, on a basis
corrected the next morning, and nothing on the slide said so.

ONE PANEL PER SESSION. The previous renderer keyed every figure by animal and drew a single "post"
column -- invisible while each animal had one post-stroke night; with two it averages a day-1 and a
day-2 brain, and for PS94 those differ more from each other than pre differs from post.

BOTH ARMS, into separate files, never compared panel-to-panel: the ALL arm scores six positions at a
fixed 1/6 chance, while the LICK-ONLY arm uses each session's preserved positions, so its chance
level moves with the animal's behaviour.

THE SMALL-LESION FAMILY (G7) IS THE SAME FIGURES over the EXCLUDED sessions. PS92/PS93 8/17 follow
the 8/16 laser that did not take. They are NOT a no-lesion control -- both animals were lesioned,
just too mildly to produce an overt deficit -- and their real value is as the within-animal
before/after control for the same animals' 8/18 sessions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfield_local import config
from wfield_local import plot_poststroke as pp
from wfield_local.paths import PathResolver

#: condition key in the JSON -> the align string the confusion figure expects
#: The lick-aligned confusion exists on the LICK-ONLY arm only -- see the guard in
#: poststroke_section_g. _render_family simply finds nothing for it on the all-trials arm.
CONFUSION_ALIGNS = (("post-cue", "cue"), ("pre-cue", "precue"), ("post-lick", "lick"))
ARMS = (("all", "ALL trials"), ("lickonly", "LICK-ONLY"))


def _sessions(rec, tag):
    return {k: v for k, v in sorted(rec.items()) if v.get("phase_tag") == tag}


def copy_behaviour_figures(out, rv=None):
    """G1a: the behaviour pipeline's own per-animal longitudinal figure, copied into the deck dir.

    Copied rather than redrawn so the deck's behavioural ground truth IS the artefact the behaviour
    pipeline produces -- if the two ever disagree that is a bug, not a styling difference.

    THE ANIMAL LIST COMES FROM THE PHASE RESOLVER. The script this replaces filtered on
    `session_phase(a, "0817") != "post"`, which is false for PS92 and PS93 -- their effective lesion
    is 8/18 -- so PS93 had no behaviour slide and the deck build reported one missing figure with
    nothing to say why.
    """
    import shutil

    rv = rv or PathResolver()
    beh = Path(rv.root("behavior_out")) / "cohort" / "by_animal"
    made = []
    for a in sorted({config.animal_of(lab) for lab in config.phase_labels("post")}):
        srcp = beh / f"{a}_across_sessions.png"
        if not srcp.exists():
            print(f"  MISSING G1a {a}: {srcp} "
                  f"(run `python -m wfield_local.spout_behavior --cohort`)", flush=True)
            continue
        dst = Path(out) / f"poststroke_G1a_behaviour_{a}.png"
        shutil.copy2(srcp, dst)
        made.append(dst)
    return made


#: G2b condition -> (arm, confusion condition, pre phase, post phase).
#:
#: The arm matters. "WITH lick" is the ENGAGED post panel, which is the lick-only arm's `post`;
#: the no-lick condition pairs `pre_nolick` with `post_nolick`, which are decoder-identical across
#: arms (both are the frozen pre-stroke-engaged decoder applied to no-lick trials), so either arm
#: serves and "all" is used to keep the pairing in one record.
G2B_CONDITIONS = (
    ("post-cue", "lickonly", "post-cue", "pre", "post"),
    ("post-lick", "lickonly", "post-lick", "pre", "post"),
    ("pre-cue WITH lick", "lickonly", "pre-cue", "pre", "post"),
    ("pre-cue NO lick", "all", "pre-cue", "pre_nolick", "post_nolick"),
)


def _recall_row(phase_rec):
    """Per-position {n, recall} from ONE confusion phase.

    The matrix is row-normalised with TRUE position on the rows, so its diagonal IS the per-position
    recall -- `crossed_confusion` says so explicitly ("its diagonal is the per-position recall table,
    so the two are one object rather than two"). Verified against the superseded day-1 JSON this
    replaces: PS94 post-cue pre reads 0.974/0.916/0.955/0.945/0.903/0.913 on n=976/957/957/931/929/921
    from both. So G2b needed no new decoding, only the lick alignment and a post-no-lick panel.
    """
    M = phase_rec["matrix"]
    ns = phase_rec["n_per_true_position"]
    pos = phase_rec["positions"]
    out = {}
    for i, nm in enumerate(pos):
        v = M[i][i]
        out[nm] = {"n": int(ns[i]), "recall": (float(v) if v is not None and v == v else float("nan"))}
    return out


def _balanced(row):
    """Mean recall over the positions the animal actually attempted (n > 0)."""
    vs = [r["recall"] for r in row.values() if r["n"] > 0 and r["recall"] == r["recall"]]
    return float(sum(vs) / len(vs)) if vs else float("nan")


def per_position_table(sub):
    """G2b's input: {animal: {condition: {pre, posts[(label, row)], pre_balanced, balanced}}}."""
    table = {}
    for cond, arm, ckey, pre_ph, post_ph in G2B_CONDITIONS:
        for lab in sorted(sub):
            animal = lab.split("_")[0]
            conf = (sub[lab].get("arms", {}).get(arm, {}).get("confusion", {}) or {}).get(ckey)
            if not conf or pre_ph not in conf or post_ph not in conf:
                continue
            rec = table.setdefault(animal, {}).setdefault(
                cond, {"pre": _recall_row(conf[pre_ph]), "posts": [], "balanced": {}})
            post_row = _recall_row(conf[post_ph])
            rec["posts"].append((lab, post_row))
            rec["balanced"][lab] = _balanced(post_row)
            rec["pre_balanced"] = _balanced(rec["pre"])
    for a in table:
        for c in table[a]:
            table[a][c]["posts"].sort(key=lambda kv: kv[0])
    return table


def _render_family(sub, out, prefix, label):
    """Every arm-dependent figure for one family: post-stroke, or the small-lesion comparison."""
    made = []
    if not sub:
        return made
    counts = {k: v["counts"] for k, v in sub.items() if v.get("counts")}
    if counts:
        made.extend(pp.fig_behaviour(
            counts, out, name=f"{prefix}_counts.png",
            suptitle=(f"{label}: which positions each session still attempts. A position with ZERO "
                      f"engaged trials has no lick-only decoding number at all, and reading one as "
                      f"a neural deficit is how the first pass went wrong. Pre-stroke bars are the "
                      f"per-session MEAN; post bars are that single session.")))

    for arm, arm_name in ARMS:
        matched = {k: {c: v["arms"][arm][c] for c in ("post-cue", "post-lick", "pre-cue")
                       if c in v.get("arms", {}).get(arm, {})}
                   for k, v in sub.items()}
        matched = {k: v for k, v in matched.items() if v}
        if matched:
            chance = {k: sub[k]["arms"][arm].get("chance", 1 / 6) for k in matched}
            made.append(pp.fig_matched(
                matched, out, chance=chance, name=f"{prefix}_matched_{arm}.png",
                suptitle=(f"{label} — FROZEN pre-stroke decoder, post arm = {arm_name}. One "
                          f"panel per session, against that animal's own pre-stroke band. "
                          + ("Six positions, chance 1/6 on every panel, so the panels ARE "
                             "comparable." if arm == "all" else
                             "Each session on ITS OWN preserved positions, so the chance line "
                             "differs between panels and accuracies must NOT be compared across "
                             "them."))))

        sim = {k: v["arms"][arm]["postcue_pattern_similarity"] for k, v in sub.items()
               if v.get("arms", {}).get(arm, {}).get("postcue_pattern_similarity")}
        if sim:
            made.append(pp.fig_similarity(
                sim, out, name=f"{prefix}_similarity_{arm}.png",
                suptitle=(f"{label} — per-position correlation between the pre- and post-stroke "
                          f"mean patterns, post arm = {arm_name}. Decoding accuracy alone cannot "
                          f"separate a weakened code from a reorganised one; this can.")))

        for cond, align in CONFUSION_ALIGNS:
            conf = {k: {align: v["arms"][arm]["confusion"][cond]} for k, v in sub.items()
                    if v.get("arms", {}).get(arm, {}).get("confusion", {}).get(cond)}
            if conf:
                # one figure per session now -- extend, not append
                made += pp.fig_confusion_alltrials(
                    conf, out, align=align, name=f"{prefix}_confusion_{align}_{arm}.png")

        made.append(pp.fig_grid(sub, out, arm=arm, name=f"{prefix}_grid_{arm}.png"))
    return made


def _render_readouts(sub, out, prefix):
    """The arm-INDEPENDENT no-lick readouts (G4, G4b, G6).

    They read the no-lick arm on purpose, which is what they are for, so they are computed and drawn
    once per session rather than once per arm.
    """
    made = []
    ident = {k: v["looks_like_which"] for k, v in sub.items() if v.get("looks_like_which")}
    if ident:
        made.append(pp.fig_identity(ident, out))
    # fig_fits_engaged indexes fits[session][align] -- it was written when the readout was computed
    # per alignment. The runner stores the pre-cue record directly, so it is nested back here rather
    # than changing a figure that reads correctly. Silently producing NOTHING is the failure this
    # avoids: the renderer's own `if m` filter swallowed the None and the figure just never appeared.
    for align in ("precue", "cue"):
        fits = {k: {align: v[f"fits_engaged_{align}"]} for k, v in sub.items()
                if v.get(f"fits_engaged_{align}")}
        if not fits:
            continue
        f = pp.fig_fits_engaged(fits, out, align=align,
                                name=f"{prefix}_fits_engaged_{align}.png")
        if f is None:
            print(f"  !! fits_engaged[{align}] produced no figure despite input for "
                  f"{sorted(fits)}", flush=True)
        made.append(f)
    rd = {k: v["impaired_nolick"] for k, v in sub.items() if v.get("impaired_nolick")}
    if rd:
        made.append(pp.fig_nolick_readout(rd, out))
    return made


def render(rec, out):
    made = []
    post = _sessions(rec, "post")
    excluded = _sessions(rec, "excluded")
    made += _render_family(post, out, "section_g", "POST-STROKE")
    made += _render_readouts(post, out, "section_g")
    made += _render_family(excluded, out, "section_g_smalllesion",
                           "SMALL-LESION COMPARISON (the laser did not take)")

    # G2b: per-position recall in the four conditions, derived from the confusion diagonals.
    ptab = per_position_table(post)
    if ptab:
        made.append(pp.fig_per_position(ptab, out, name="section_g_G2b_per_position.png"))

    # G2c: the grid over BOTH families together. fig_grid colours on phase_tag == "excluded"
    # precisely so PS92/PS93's ineffective-lesion 8/17 sits as a grey square beside their effective
    # day 1 -- a within-animal before/after control, same rig, one day apart. Rendering the families
    # separately drops that pairing: section_g_grid_* holds only the 10 post sessions and
    # section_g_smalllesion_grid_* only the 2 excluded ones, so NEITHER is the figure the G2c slide
    # describes. Until 2026-08-20 the deck used a scratchpad-era poststroke_grid.png that no step
    # rewrote, so the headline four-animal slide silently stayed at its 8/19 10:17 content.
    both = {**post, **excluded}
    if both:
        for arm, _ in ARMS:
            made.append(pp.fig_grid(both, out, arm=arm,
                                    name=f"section_g_grid_withcontrol_{arm}.png"))
    return [m for m in made if m]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    src = args.src or Path(PathResolver().root("figures_working"))
    out = args.output or src
    for q in copy_behaviour_figures(out):
        print(f"  copied {Path(q).name}", flush=True)
    p = Path(src) / "section_g.json"
    if not p.exists():
        print(f"  SKIP: {p} not computed yet "
              f"(run `python -m wfield_local.poststroke_section_g`)", flush=True)
        return 0
    rec = json.load(open(p))
    print(f"\n=== {len(rec)} sessions", flush=True)
    for q in render(rec, out):
        print(f"  wrote {Path(q).name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render EVERY Section G figure from `section_g.json`, one panel per POST-STROKE SESSION.

Reads the JSON rather than recomputing, so a figure and the number quoted beside it cannot drift
apart: if they disagree, one of the two files is stale and that is visible rather than silent.

This and `poststroke_section_g` replace four scratchpad scripts. One opened with
`for an in ("PS94", "PS95")` and another with `if config.session_phase(a, "0817") != "post"`, so PS92
and PS93 were absent from every figure downstream and PS93 had no behavior slide at all. Twelve of
the deck's figures were still drawn from files those scripts wrote on 2026-08-18, on a basis
corrected the next morning, and nothing on the slide said so.

ONE PANEL PER SESSION. The previous renderer keyed every figure by animal and drew a single "post"
column -- invisible while each animal had one post-stroke night; with two it averages a day-1 and a
day-2 brain, and for PS94 those differ more from each other than pre differs from post.

BOTH ARMS, into separate files, never compared panel-to-panel: the ALL arm scores six positions at a
fixed 1/6 chance, while the LICK-ONLY arm uses each session's preserved positions, so its chance
level moves with the animal's behavior.

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
from wfield_local import trial_set_text as tst
from wfield_local.paths import PathResolver


# ------------------------------------------------------------------------------------------- the trial-set sentences
# Priya 11-09: EVERY figure states exactly which trials it includes / excludes. Section G's figures draw DIFFERENT row
# sets (read off `poststroke_compare`, which computed section_g.json): the frozen-decoder figures (matched, confusion,
# G2b) use the camera WORKING rows for the pre band / lick-only arm under the camera grouping; the within-session
# figures (grid, retained) and the pattern similarity use the lick-source ENGAGED rows (`recoding_test`,
# `pattern_similarity`); the ALL arm's post side is every cued trial; the G1-G3 readouts train camera WORKING vs raw
# NO RESPONSE (or DAQ lick vs no-lick). Each figure group is drawn inside `tst.active(None, text=...)` with its own
# sentence, so the savefig hook stamps it.

def _core(s: str) -> str:
    """A `trial_set_text.sentence` without its leading "trials: "."""
    return s.split(": ", 1)[1] if s.startswith("trials: ") else s


def lick_rows() -> str:
    """The pool's ENGAGED rows: a lick (the run's lick source) within max_rt of the cue (`tst.sentence("lick")`)."""
    return _core(tst.sentence("lick")).split("; excluded:", 1)[0]


def nolick_rows() -> str:
    """The pool's NO-LICK rows (XU): no lick of the run's lick source within max_rt of the cue."""
    return f"no-lick trials (no {tst.lick_text()} within {tst._max_rt_s()} of the cue)"


def working_rows() -> str:
    """The camera WORKING rows (`tst.sentence("working")`), in parentheses-ready form."""
    return f"camera WORKING trials ({_core(tst.sentence('working'))})"


#: the ALL arm's post side (`decode_matched` / `recoding_test` / `pattern_similarity` / `crossed_confusion`)
def all_post() -> str:
    return ("EVERY cued trial of the session at all six spouts (lick + no-lick trials; no camera-class, working-period "
            "or quit-tail cut)")


def _min_post() -> int:
    try:
        from wfield_local import poststroke_compare as pc
        return int(pc.MIN_POST)
    except Exception:                                                  # noqa: BLE001 - a caption never kills a figure
        return 20


def camera_grouping(sub=None) -> bool:
    """Did the run that wrote these records use section G's CAMERA grouping (`poststroke_compare._camera_groups`)?
    Read off the records themselves (decode_matched ``pre_band_rows``, crossed_confusion ``groups``), so a replot of an
    older json captions what that json holds; else from the run's switches."""
    for r in (sub or {}).values():
        for a in ((r or {}).get("arms") or {}).values():
            for v in (a or {}).values():
                if isinstance(v, dict) and "pre_band_rows" in v:
                    return str(v["pre_band_rows"]).startswith("camera")
            for c in ((a or {}).get("confusion") or {}).values():
                if isinstance(c, dict) and "groups" in c:
                    return c["groups"] == "camera"
    try:
        from wfield_local import poststroke_compare as pc
        from wfield_local import trial_classes as tc
        from wfield_local import working_trials as wt
        return (pc.section_g_groups() == "camera" and tc.camera_gate_on()
                and wt.is_camera(tc.working_selection()))
    except Exception:                                                  # noqa: BLE001
        return False


def _preserved(cam: bool) -> str:
    return (f"the session's preserved spouts (>= {_min_post()} "
            + ("camera WORKING trials there)" if cam else "lick trials there)"))


def counts_sentence() -> str:
    """G1 counts (`fig_behaviour`): every cued trial, split per spout into lick / no-lick trials."""
    return (f"trials: every cued trial of each session, per spout, split into {lick_rows()} and {nolick_rows()}; "
            "nothing excluded (no camera-class, working-period or quit-tail cut)")


def matched_sentence(arm: str, cam: bool) -> str:
    """G2 matched (`decode_matched`): the pre band and post rows per arm."""
    pre = working_rows() if cam else lick_rows()
    if arm == "all":
        return f"trials: pre band = pre-stroke {pre} at all six spouts; post = {all_post()}"
    return (f"trials: post-cue / pre-cue = {pre} at {_preserved(cam)}, pre band and post alike; post-lick = "
            f"{lick_rows()} at the spouts with >= {_min_post()} lick trials, pre band and post alike; excluded: "
            + ("every non-working trial" if cam else "no-lick trials") + " and every other spout")


def similarity_sentence(arm: str, cam: bool) -> str:
    """G5 pattern similarity (`pattern_similarity`): lick-source engaged rows pre; the arm's post rows."""
    if arm == "all":
        return f"trials: pre = pre-stroke {lick_rows()} per spout; post = {all_post()}"
    return (f"trials: pre and post = {lick_rows()} at {_preserved(cam)}; excluded: no-lick trials and "
            "every other spout")


def confusion_sentence(arm: str, align: str, cam: bool) -> str:
    """G3b confusion (`crossed_confusion`) panels: PRE, PRE no-lick, POST, POST no-lick (no no-lick panels post-lick;
    the lick window has no camera grouping)."""
    cam = cam and align != "lick"
    rows = working_rows() if cam else lick_rows()
    post = all_post() if arm == "all" else f"the session's {rows}, all six spouts"
    parts = [f"PRE = pre-stroke {rows}, all six spouts"]
    if align != "lick":
        parts.append(f"PRE no-lick = pre-stroke {nolick_rows()}")
    parts.append(f"POST = {post}")
    if align != "lick":
        parts.append(f"POST no-lick = the session's {nolick_rows()}")
    return "trials: " + "; ".join(parts)


def within_sentence(arm: str, cam: bool | None = None) -> str:
    """Within-session decoding (`recoding_test`: grid G2c, retained): lick-source engaged rows for the pre band."""
    cam = camera_grouping() if cam is None else cam
    if arm == "all":
        return f"trials: pre band = pre-stroke {lick_rows()} at all six spouts; post = {all_post()}"
    return (f"trials: pre band and post = {lick_rows()} at {_preserved(cam)} (post-lick: spouts with >= "
            f"{_min_post()} lick trials); excluded: no-lick trials and every other spout")


def per_position_sentence(cam: bool) -> str:
    """G2b (`per_position_table`: the confusion diagonals of the LICK-ONLY arm, plus the no-lick panels)."""
    rows = working_rows() if cam else lick_rows()
    return (f"trials: post-cue and pre-cue WITH lick = {rows} (pre-stroke sessions and the post session); post-lick = "
            f"{lick_rows()}; pre-cue NO lick = {nolick_rows()} (pre-stroke and post)")


def g13_sentence(records=None) -> str:
    """G4 identity / G4b fits-engaged (`looks_like_which`, `fits_engaged_distribution`): the discriminator's classes
    and the scored rows, from the records' ``train_classes`` (camera vs DAQ)."""
    cam = any(((r or {}).get("train_classes") or {}).get("groups") == "camera" for r in (records or []))
    if not records:
        cam = camera_grouping()
    if cam:
        return (f"trials: discriminator trained on pre-stroke {working_rows()} vs camera NO RESPONSE trials (raw class "
                "no_response: no lick seen on camera, within the working period), position-balanced at the session's "
                "preserved spouts; scored: the post session's no-response trials; never trained on: unknown, "
                "jaw-only, quit, stopped, stopped-licking")
    return (f"trials: discriminator trained on pre-stroke {lick_rows()} vs {nolick_rows()}, position-balanced at the "
            "session's preserved spouts; scored: the post session's no-lick trials")


def nolick_readout_sentence() -> str:
    """G6 (`impaired_nolick_readout`): the post session's no-lick trials, preserved vs impaired spouts."""
    return (f"trials: the post session's {nolick_rows()}, split into preserved spouts and impaired spouts (the rest); "
            "excluded: lick trials")


def _act(text_fn, *a):
    """`tst.active` with a sentence built by ``text_fn(*a)`` (a failure to build it never costs the figure)."""
    try:
        text = text_fn(*a)
    except Exception as ex:                                            # noqa: BLE001
        print(f"  [trial set] no caption ({type(ex).__name__}: {ex})", flush=True)
        text = None
    return tst.active(None, text=text) if text else tst.active(None)

#: condition key in the JSON -> the align string the confusion figure expects
#: The lick-aligned confusion exists on the LICK-ONLY arm only -- see the guard in
#: poststroke_section_g. _render_family simply finds nothing for it on the all-trials arm.
CONFUSION_ALIGNS = (("post-cue", "cue"), ("pre-cue", "precue"), ("post-lick", "lick"))
ARMS = (("all", "ALL trials"), ("lickonly", "LICK-ONLY"))


def _sessions(rec, tag):
    return {k: v for k, v in sorted(rec.items()) if v.get("phase_tag") == tag}


def copy_behaviour_figures(out, rv=None):
    """G1a: the behavior pipeline's own per-animal longitudinal figure, copied into the deck dir.

    Copied rather than redrawn so the deck's behavioral ground truth IS the artifact the behavior
    pipeline produces -- if the two ever disagree that is a bug, not a styling difference.

    THE ANIMAL LIST COMES FROM THE PHASE RESOLVER. The script this replaces filtered on
    `session_phase(a, "0817") != "post"`, which is false for PS92 and PS93 -- their effective lesion
    is 8/18 -- so PS93 had no behavior slide and the deck build reported one missing figure with
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

    The matrix is row-normalized with TRUE position on the rows, so its diagonal IS the per-position
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
        # `v != v` is the NaN test; ruff reads it as a self-comparison, hence the noqa.
        ok = v is not None and v == v                                        # noqa: PLR0124
        out[nm] = {"n": int(ns[i]), "recall": (float(v) if ok else float("nan"))}
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
    for conds in table.values():
        for rec_c in conds.values():
            rec_c["posts"].sort(key=lambda kv: kv[0])
    return table


def _render_family(sub, out, prefix, label):
    """Every arm-dependent figure for one family: post-stroke, or the small-lesion comparison."""
    made = []
    if not sub:
        return made
    cam = camera_grouping(sub)
    counts = {k: v["counts"] for k, v in sub.items() if v.get("counts")}
    if counts:
        with _act(counts_sentence):
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
            with _act(matched_sentence, arm, cam):
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
            with _act(similarity_sentence, arm, cam):
                made.append(pp.fig_similarity(
                    sim, out, name=f"{prefix}_similarity_{arm}.png",
                    suptitle=(f"{label} — per-position correlation between the pre- and post-stroke "
                              f"mean patterns, post arm = {arm_name}. Decoding accuracy alone cannot "
                              f"separate a weakened code from a reorganized one; this can.")))

        for cond, align in CONFUSION_ALIGNS:
            conf = {k: {align: v["arms"][arm]["confusion"][cond]} for k, v in sub.items()
                    if v.get("arms", {}).get(arm, {}).get("confusion", {}).get(cond)}
            if conf:
                # one figure per session now -- extend, not append
                # arm_name is what LABELS the post panels. Passing only the filename left every
                # LICK-ONLY figure captioned "ALL trials" (fixed 2026-08-24).
                with _act(confusion_sentence, arm, align, cam):
                    made += pp.fig_confusion_alltrials(
                        conf, out, align=align, name=f"{prefix}_confusion_{align}_{arm}.png",
                        arm_name=arm_name)

        with _act(within_sentence, arm, cam):
            made.append(pp.fig_grid(sub, out, arm=arm, name=f"{prefix}_grid_{arm}.png"))
    return made


def _render_readouts(sub, out, prefix, *, named_only=False):
    """The arm-INDEPENDENT no-lick readouts (G4, G4b, G6).

    They read the no-lick arm on purpose, which is what they are for, so they are computed and drawn
    once per session rather than once per arm.

    ``named_only`` RENDERS ONLY THE FIGURES THAT ACCEPT A FILENAME, and exists so a second family
    can be rendered without destroying the first's output. `fig_identity` (G4) and
    `fig_nolick_readout` (G6) take no `name` and write FIXED paths that carry no prefix, so calling
    this twice would have the small-lesion family silently overwrite the post-stroke family's two
    headline slides -- one output path claimed by two units, which is the collision the grant
    render's own guard exists to catch and which nothing checks here. The small-lesion family
    therefore takes the fits-engaged pair only, which is the one G7d places.
    """
    made = []
    # G4 PRIMARY (coordinator 11-04): the camera-class discriminator, with the DAQ discriminator as a labelled robustness
    # panel (`g4_figures`; figdata + --plot-only there). Prefixed, so the two families never share a path. The DAQ G4
    # figures below are kept as the robustness set.
    if any(v.get("g4_camclass_cue") or v.get("g4_camclass_precue") for v in sub.values()):
        from wfield_local import g4_figures
        made.append(g4_figures.render(sub, out, prefix))     # its own sentence (`g4_figures.trial_set_sentence`)
    ident = {k: v["looks_like_which"] for k, v in sub.items() if v.get("looks_like_which")}
    if ident and not named_only:
        with _act(g13_sentence, list(ident.values())):
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
        with _act(g13_sentence, [r[align] for r in fits.values()]):
            f = pp.fig_fits_engaged(fits, out, align=align,
                                    name=f"{prefix}_fits_engaged_{align}.png")
        if f is None:
            print(f"  !! fits_engaged[{align}] produced no figure despite input for "
                  f"{sorted(fits)}", flush=True)
        made.append(f)
    rd = {k: v["impaired_nolick"] for k, v in sub.items() if v.get("impaired_nolick")}
    if rd and not named_only:
        with _act(nolick_readout_sentence):
            made.append(pp.fig_nolick_readout(rd, out))
    return made


def render(rec, out, *, json_path=None, suffix="", note=""):
    made = []
    post = _sessions(rec, "post")
    excluded = _sessions(rec, "excluded")
    # EVERY NAME CARRIES ``suffix`` (D9 queue audit 11-08): it used to reach only the retained figure, so a second arm
    # rendered into the same folder (the contact-only supplement, --suffix _contactonly) overwrote the primary figures.
    # suffix "" leaves every name exactly as before.
    made += _render_family(post, out, f"section_g{suffix}", "POST-STROKE")
    # G4 / G6 write FIXED paths (no name argument): with a suffix only the named readouts are drawn, so the second arm
    # cannot overwrite the primary's two fixed-path figures (they are arm-independent no-lick readouts).
    made += _render_readouts(post, out, f"section_g{suffix}", named_only=bool(suffix))
    made += _render_family(excluded, out, f"section_g_smalllesion{suffix}",
                           "SMALL-LESION COMPARISON (the laser did not take)")
    # THE FAILED-LASER FITS-ENGAGED PAIR (G7d), which until 2026-09-09 NO STEP PRODUCED. The deck
    # placed a pair of files hand-made on 2026-08-18 and justified freezing them on the grounds that
    # "this comparison is permanently PS92/PS93 on 8/17, so its content cannot change". The sessions
    # cannot change; the ANALYSIS did. `fits_engaged` is computed on ENGAGED trials, and the
    # engagement gate was replaced twice since -- position-blind, then reference-restricted and
    # backdated (880e6bd) -- so those figures were drawn against a gate that no other figure in the
    # deck still uses. A frozen figure is only safe when nothing upstream of it moves.
    made += _render_readouts(excluded, out, f"section_g_smalllesion{suffix}", named_only=True)

    # G2b: per-position recall in the four conditions, derived from the confusion diagonals.
    ptab = per_position_table(post)
    if ptab:
        with _act(per_position_sentence, camera_grouping(post)):
            made.append(pp.fig_per_position(ptab, out, name=f"section_g_G2b_per_position{suffix}.png"))

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
            with _act(within_sentence, arm, camera_grouping(both)):
                made.append(pp.fig_grid(both, out, arm=arm,
                                        name=f"section_g_grid_withcontrol_{arm}{suffix}.png"))

    # RETAINED ABOVE-CHANCE FRACTION (Priya 11-07): within-session accuracy as (post - c) / (pre - c)
    # against the animal's pre band on the SAME positions, so LICK-ONLY sessions scored 4-, 5- and
    # 6-way share one scale. Both families in one figure; excluded sessions are drawn grey.
    from wfield_local import section_g_retained
    made += section_g_retained.render(rec, out, suffix=suffix, json_path=json_path, note=note)
    return [m for m in made if m]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None,
                    help="read this section_g.json instead of the run's own (e.g. a contact-only "
                         "supplement run); needs --output")
    ap.add_argument("--suffix", default="",
                    help="appended to the retained-fraction figure / CSV stem (e.g. _contactonly)")
    ap.add_argument("--note", default="",
                    help="short label for the retained figure's title / printed table (e.g. 'old gate')")
    ap.add_argument("--only", choices=("retained",), default=None,
                    help="render only this figure (retained = the n-corrected retained fraction)")
    args = ap.parse_args(argv)
    if args.only == "retained":
        from wfield_local import section_g_retained
        if args.json is None or args.output is None:
            ap.error("--only retained needs --json and --output")
        rec = json.loads(args.json.read_text(encoding="utf-8"))
        for q in section_g_retained.render(rec, args.output, suffix=args.suffix,
                                           json_path=args.json, note=args.note):
            print(f"  wrote {Path(q).name}", flush=True)
        return 0
    from wfield_local import run_mode  # tagged runs never overwrite default figures (review H2)
    src = run_mode.routed_output(args.src or Path(PathResolver().root("figures_working")), make=False, guard=False)
    out = args.output or src
    for q in copy_behaviour_figures(out):
        print(f"  copied {Path(q).name}", flush=True)
    from wfield_local import run_mode   # the JSON of THIS run's mode (rebuild 11-03)
    p = args.json or Path(src) / run_mode.keyed_name("section_g.json")
    if not p.exists():
        print(f"  SKIP: {p} not computed yet "
              f"(run `python -m wfield_local.poststroke_section_g`)", flush=True)
        return 0
    rec = json.loads(Path(p).read_text(encoding="utf-8"))
    print(f"\n=== {len(rec)} sessions", flush=True)
    for q in render(rec, out, json_path=p, suffix=args.suffix, note=args.note):
        print(f"  wrote {Path(q).name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

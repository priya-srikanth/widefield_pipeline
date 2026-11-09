"""Section G: RETAINED ABOVE-CHANCE FRACTION of within-session decoding, n-corrected (Priya 2026-11-07).

WHY A RATIO. The LICK-ONLY arm scores each post-stroke session on ITS OWN preserved positions, so its
chance level is 1/n with n = 4, 5 or 6 depending on what the animal still attempts. A raw within-session
accuracy of 0.6 is then a different achievement on a 4-way problem (chance 0.25) than on a 6-way one
(chance 0.167), and `fig_matched` / `fig_grid` say so in their captions: their panels must not be
compared. `recoding_test` already scores the animal's pre-stroke sessions on the SAME positions, so the
band and the post value share one chance level. Dividing the post margin above chance by the pre margin
above chance puts every session on one scale:

    retained = (post_within_acc - c) / (pre_band_mean - c)      c = 1 / n_positions of THAT record

1 = the session decodes as far above chance as the animal's own pre-stroke sessions on those positions;
0 = at chance. The secondary column ``dnorm`` is the difference of chance-normalized accuracies,
``(post - c)/(1 - c) - (pre - c)/(1 - c)``, which does not blow up when the pre margin is small.

Reads ``section_g.json`` (written by `poststroke_section_g`), so the numbers here and on the other
Section G figures come from one file. Keys used, per session record ``rec[label]``:
``phase_tag``, ``animal``, ``arms[arm]["chance"]``, and
``arms[arm]["<cond> within-session"]`` -> ``n_positions``, ``chance``, ``post_arm``,
``within_pre_band`` {mean, sd, min, max, n}, ``post[label]`` {within_accuracy, z, inside_pre_range}.

No CI is drawn: the quantity is per session against a band MEAN, and the per-animal x epoch table is
printed (CLAUDE.md rule 8) so any later pooled test starts from what each animal shows.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

ARMS = (("all", "ALL trials"), ("lickonly", "LICK-ONLY"))
CONDS = ("post-cue", "pre-cue", "post-lick")
EPOCH_ORDER = ("acute", "subacute", "chronic", "excluded")
COLUMNS = ("animal", "label", "phase_tag", "epoch", "day", "arm", "cond", "n_positions", "chance",
           "pre_mean", "pre_n", "pre_margin", "post_acc", "retained", "dnorm", "z", "inside_pre_range")
EXCLUDED_GREY = "0.6"


def retained_fraction(post, pre, chance):
    """(post - c) / (pre - c); NaN when the pre band is not above chance (nothing to retain)."""
    margin = pre - chance
    if not (margin > 0):
        return float("nan")
    return (post - chance) / margin


def dnorm(post, pre, chance):
    """(post - c)/(1 - c) - (pre - c)/(1 - c): change in chance-normalized accuracy."""
    return ((post - chance) - (pre - chance)) / (1.0 - chance)


def _day_epoch(label, phase_tag):
    """(days since stroke, epoch) from `epochs` (one definition); excluded sessions get epoch 'excluded'."""
    try:
        from wfield_local import epochs
        day = epochs.days_since_stroke(label)
        ep = epochs.epoch_of(label)
    except Exception:                                                  # noqa: BLE001 (synthetic labels in tests)
        day, ep = None, None
    if phase_tag == "excluded":
        ep = "excluded"
    return day, ep


def retained_table(rec):
    """Tidy rows, one per (session, arm, cond) that has a within-session record for that session.

    Sorted by (arm, cond, animal, label) so the CSV and any pool built from it have a defined order."""
    rows = []
    for label in sorted(rec):
        r = rec[label]
        tag = r.get("phase_tag")
        if tag not in ("post", "excluded"):
            continue
        animal = r.get("animal") or label.split("_")[0]
        day, ep = _day_epoch(label, tag)
        for arm, _ in ARMS:
            a = (r.get("arms") or {}).get(arm) or {}
            for cond in CONDS:
                w = a.get(f"{cond} within-session")
                if not w or label not in (w.get("post") or {}):
                    continue
                band = w["within_pre_band"]
                p = w["post"][label]
                c = float(w.get("chance", a.get("chance")))
                if a.get("chance") is not None and abs(float(a["chance"]) - c) > 1e-9:
                    print(f"  !! {label} {arm} {cond}: within-session chance {c:.3f} != arm chance "
                          f"{a['chance']:.3f}; using the within-session record's", flush=True)
                pre, post = float(band["mean"]), float(p["within_accuracy"])
                rows.append({"animal": animal, "label": label, "phase_tag": tag, "epoch": ep, "day": day,
                             "arm": arm, "cond": cond, "n_positions": int(w.get("n_positions") or round(1 / c)),
                             "chance": c, "pre_mean": pre, "pre_n": int(band.get("n", 0)),
                             "pre_margin": pre - c, "post_acc": post,
                             "retained": retained_fraction(post, pre, c), "dnorm": dnorm(post, pre, c),
                             "z": float(p.get("z", float("nan"))),
                             "inside_pre_range": bool(p.get("inside_pre_range"))})
    arm_i = {a: i for i, (a, _) in enumerate(ARMS)}
    cond_i = {c: i for i, c in enumerate(CONDS)}
    rows.sort(key=lambda d: (arm_i[d["arm"]], cond_i[d["cond"]], d["animal"], d["label"]))
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for d in rows:
            w.writerow({k: ("" if d[k] is None else d[k]) for k in COLUMNS})
    return path


def _median(vs):
    vs = sorted(v for v in vs if v == v)
    if not vs:
        return float("nan")
    m = len(vs) // 2
    return vs[m] if len(vs) % 2 else 0.5 * (vs[m - 1] + vs[m])


def animal_epoch_table(rows):
    """{(arm, cond): {animal: {epoch: (median retained, n sessions)}}}."""
    out = {}
    for d in rows:
        cell = out.setdefault((d["arm"], d["cond"]), {}).setdefault(d["animal"], {})
        cell.setdefault(d["epoch"] or "none", []).append(d["retained"])
    return {k: {an: {ep: (_median(v), len(v)) for ep, v in eps.items()} for an, eps in sorted(t.items())}
            for k, t in out.items()}


def print_animal_epoch_table(rows, header=""):
    """CLAUDE.md rule 8: the per-animal x epoch view, printed before anything is pooled."""
    tab = animal_epoch_table(rows)
    eps = [e for e in EPOCH_ORDER if any(e in v for t in tab.values() for v in t.values())]
    lines = [f"=== retained above-chance fraction, median (n sessions) per animal x epoch {header}".rstrip()]
    for arm, arm_name in ARMS:
        for cond in CONDS:
            t = tab.get((arm, cond))
            if not t:
                continue
            lines.append(f"-- {arm_name} | {cond}")
            lines.append("   animal  " + "".join(f"{e:>16}" for e in eps))
            for an, cells in t.items():
                lines.append(f"   {an:<7} " + "".join(
                    (f"{cells[e][0]:>11.2f} ({cells[e][1]:>2})" if e in cells else f"{'-':>16}") for e in eps))
    txt = "\n".join(lines)
    print(txt, flush=True)
    return txt


def design_text(json_path):
    """Lick source / selection from the run's design sidecar beside the json, else 'per the run's gate'."""
    p = Path(json_path)
    side = p.with_name(p.name.replace("section_g", "section_g_design", 1))
    if side.exists():
        try:
            d = json.loads(side.read_text(encoding="utf-8"))
            return (f"selection {d.get('model_trials', '?')}, groups = {d.get('groups', '?')} "
                    f"({'camera lick classes' if d.get('groups') == 'camera' else 'DAQ licks'})")
        except Exception:                                              # noqa: BLE001
            pass
    return "lick source / gate per the run's gate (no design sidecar beside the json)"


def _offset(animal, animals, width=0.5):
    """Fixed per-animal x offset (days) so same-day sessions of different animals do not stack."""
    k = sorted(animals).index(animal)
    n = len(animals)
    return 0.0 if n < 2 else (k / (n - 1) - 0.5) * width


def fig_retained(rows, out, *, suffix="", json_path=None, note=""):
    """Rows = arms (ALL, LICK-ONLY), columns = cond; x = days since stroke, y = retained fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import cat_points as cp
    from wfield_local import epoch_figures as ef
    from wfield_local import sig_marks
    from wfield_local.figure_layout import save_png_svg

    animals = sorted({d["animal"] for d in rows})
    fin = [d["retained"] for d in rows if d["retained"] == d["retained"]]
    lo = min([-0.2, *fin]) - 0.1
    hi = max([1.2, *fin]) + 0.1
    days = [d["day"] for d in rows if d["day"] is not None]
    xmax = (max(days) if days else 1) + 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharey=True, sharex=True)
    for i, (arm, arm_name) in enumerate(ARMS):
        for j, cond in enumerate(CONDS):
            ax = axes[i, j]
            sub = [d for d in rows if d["arm"] == arm and d["cond"] == cond]
            if not sub:
                ax.text(0.5, 0.5, f"no {arm_name} {cond} record\n(lick-aligned exists on the\nLICK-ONLY arm only)"
                        if cond == "post-lick" else f"no {arm_name} {cond} record",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.4")
                ax.axis("off")
                continue
            ax.axhline(1.0, color="0.25", ls="--", lw=1, zorder=1)
            ax.axhline(0.0, color="0.25", ls=":", lw=1, zorder=1)
            for d in sub:
                if d["day"] is None or d["retained"] != d["retained"]:
                    continue
                col = EXCLUDED_GREY if d["epoch"] == "excluded" else ef.epoch_dot_color(d["epoch"])
                ax.plot(d["day"] + _offset(d["animal"], animals), d["retained"], ls="none",
                        marker=cp.animal_marker(d["animal"]), mfc="none", mec=col, mew=1.1, ms=6, zorder=3)
            # pooled epoch median: a short bar over that epoch's day range (no line across epochs)
            for ep in ("acute", "subacute", "chronic"):
                es = [d for d in sub if d["epoch"] == ep and d["day"] is not None and d["retained"] == d["retained"]]
                if not es:
                    continue
                med = _median([d["retained"] for d in es])
                x0, x1 = min(d["day"] for d in es) - 0.5, max(d["day"] for d in es) + 0.5
                ax.plot([x0, x1], [med, med], color=ef.epoch_dot_color(ep), lw=2.4, alpha=0.8, zorder=2,
                        solid_capstyle="butt")
            ns = sorted({d["n_positions"] for d in sub})
            ax.set_title(f"{arm_name} | {cond}  (n positions {'/'.join(map(str, ns))})", fontsize=10)
            ax.set_xlim(-1, xmax)
            ax.set_ylim(lo, hi)
            if j == 0:
                ax.set_ylabel(f"{arm_name}\nretained above-chance fraction", fontsize=10)
            if i == 1:
                ax.set_xlabel("days since stroke", fontsize=10)
            ax.tick_params(labelsize=9, labelbottom=True)
            ax.spines["right"].set_visible(False)
    # legends: shape = animal, colour = epoch, reference lines
    eps = [e for e in EPOCH_ORDER if any(d["epoch"] == e for d in rows)]
    eh = [Line2D([], [], ls="none", marker="o", mfc="none", mew=1.1, ms=6,
                 mec=(EXCLUDED_GREY if e == "excluded" else ef.epoch_dot_color(e)),
                 label=("excluded (small lesion, day 0)" if e == "excluded" else e)) for e in eps]
    rh = [Line2D([], [], color="0.25", ls="--", lw=1, label="1 = animal's pre-stroke level"),
          Line2D([], [], color="0.25", ls=":", lw=1, label="0 = chance (1/n)"),
          Line2D([], [], color="0.4", lw=2.4, label="epoch median (pooled sessions)")]
    fig.legend(handles=ef._shape_handles(animals) + eh + rh, loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    design = design_text(json_path) if json_path else "per the run's gate"
    title = (
        "Section G within-session decoding: RETAINED ABOVE-CHANCE FRACTION = (post − c) / (pre − c), "
        "c = 1/n of that session's scored positions" + (f"  [{note}]" if note else "") + "\n"
        f"Window = panel column (post-cue / pre-cue / post-lick). {design}. "
        "Trial set: ALL arm = post session all trials at all 6 positions; LICK-ONLY = engaged trials on the "
        "session's preserved positions.\nBaseline = the animal's pre-stroke within-session band (mean) on the SAME "
        "positions (engaged trials). Correction = ratio to the pre margin above chance. "
        "Points: one session, shape = animal, color = epoch; x offset per animal ±0.25 d.")

    title = "\n".join(textwrap.fill(par, 175) for par in title.split("\n"))
    fig.suptitle(title, fontsize=9.5, y=0.995)
    sig_marks.reserve_headroom(fig)      # hides the top spine on every panel (no marks: no test is computed)
    fig.tight_layout(rect=(0, 0.06, 1, 1))     # tight_layout already makes room for the suptitle
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    q = out / f"section_g_retained{suffix}.png"
    save_png_svg(fig, q, dpi=150)
    plt.close(fig)
    return q


def trial_set_sentence(rec=None) -> str:
    """The figure's trial set (Priya 11-09): both arms' within-session (`recoding_test`) rows, ALL and LICK-ONLY
    (`section_g_figures.within_sentence`; camera grouping read off ``rec``)."""
    from wfield_local import section_g_figures as sgf
    from wfield_local import trial_set_text as tst
    cam = sgf.camera_grouping(rec)
    return tst.union(f"ALL arm: {sgf._core(sgf.within_sentence('all', cam))}",
                     f"LICK-ONLY arm: {sgf._core(sgf.within_sentence('lickonly', cam))}", unit="trials")


def render(rec, out, *, suffix="", json_path=None, note=""):
    """Table (printed + CSV sidecar) and figure. Returns [png]."""
    from wfield_local import trial_set_text as tst
    from wfield_local.figure_layout import sidecar
    rows = retained_table(rec)
    if not rows:
        print("  retained: no within-session records", flush=True)
        return []
    print_animal_epoch_table(rows, header=f"[{note}]" if note else "")
    try:
        text = trial_set_sentence(rec)
    except Exception as ex:                                            # noqa: BLE001 - a caption never kills a figure
        print(f"  [trial set] no caption ({type(ex).__name__}: {ex})", flush=True)
        text = None
    with tst.active(None, text=text):
        q = fig_retained(rows, out, suffix=suffix, json_path=json_path, note=note)
    write_csv(rows, sidecar(q, ".csv"))
    return [q]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", type=Path, help="a section_g.json (any working selection)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--suffix", default="", help="appended to output file stems, e.g. _contactonly")
    ap.add_argument("--note", default="", help="short label for the suptitle / printed table, e.g. 'old gate'")
    a = ap.parse_args(argv)
    rec = json.loads(a.json.read_text(encoding="utf-8"))
    for q in render(rec, a.output, suffix=a.suffix, json_path=a.json, note=a.note):
        print(f"  wrote {q}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

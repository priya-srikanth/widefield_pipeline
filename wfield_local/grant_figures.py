"""GRANT FIGURES — a small, self-contained set for a progress report and a new application.

    python -m wfield_local.grant_figures [--output <dir>] [--only 1 1b 2 2b 3a 3b 4 5]

Priya, 2026-08-24. Deliberately NOT deck figures: the deck exists to be interrogated and carries every
caveat on the slide, which is right there and wrong here. These are meant to be read in ten seconds by
someone who has not been in the weeds, so each one makes ONE point, and the caveats live in this
docstring and in DECISIONS.md rather than on the axes.

WHAT IS SHOWN

  1  BEHAVIOUR at all six spout positions, per animal, against days-from-lesion. Engaged hit rate --
     the "stopped" (terminal quit) trials are excluded, which is what `hit_rate` in the per-position
     metrics CSV already means -- with Wilson CIs. This is the deficit the rest of the work is about.

  2  PRE-STROKE CROSS-SESSION DECODING in the shared joint-LocaNMF basis, ENL / cue / lick. Each bar
     is leave-one-session-out accuracy pooled over the curated pre-stroke sessions, the error bar is
     the 95% CI across HELD-OUT SESSIONS (not across trials -- the session is the unit that
     generalisation is claimed over), and every held-out session is plotted as a point. Chance is
     1/6.

  3a POST-STROKE CODING RETAINED, per animal, per window, over days. y = the projection of that
     position's own trials onto its own pre-stroke coding direction, pole-normalised so 1.0 = the
     pre-stroke lick signature and 0 = the other positions. Error bars are SEM over trials. This is
     the "how much of the normal code is left" view.

  3b FROZEN vs WITHIN-SESSION DECODING, per animal, per window, over days. The frozen pre-stroke
     decoder asks whether the OLD code still reads out; a decoder trained on the post-stroke session
     itself asks whether position information is present AT ALL. Frozen falling while within-session
     holds up is reorganisation rather than loss -- the two lines and the gap between them are the
     point. Error bars are binomial 95% CIs on each session's own trial count.

EXCLUSIONS, and they are not cosmetic
  * PS92_0817 and PS93_0817 are dropped everywhere. They follow the 8/16 laser that did NOT take, so
    they are neither baseline nor post-stroke, and `config.session_phase` already labels them
    "excluded" -- this module asks the config rather than hardcoding dates.
  * PS94/PS95 were lesioned 2026-08-16 and PS92/PS93 2026-08-17, so DAY-FROM-LESION is per animal
    (`config.stroke_date`). Plotting against calendar date would misalign the cohort by a day.

WHAT THESE FIGURES DO NOT SAY, kept here so it is not lost when they are pasted into a document:
  * "Miss" is defined by SPOUT CONTACT, so an off-target lick counts as a miss in panel 1 and sits in
    the miss class in panel 3. The DAQ cannot distinguish "did not try" from "tried and missed".
  * Panel 3a's y can exceed 1.0. The projection rises either because a trial points more along the
    direction or because it sits further from the session centroid; above-1 values mean "at least
    intact", not "better than pre-stroke".
  * Panel 2 is PRE-STROKE only and is a capability claim (we can decode), not a lesion result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local import config
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
POS = ["far_R", "far_center", "far_L", "close_R", "close_center", "close_L"]
#: colour = ring, marker/linestyle = side -- the convention the behaviour figures already use, so a
#: reader who has seen those does not have to relearn it here.
POS_STYLE = {
    "far_R":        ("#b2182b", "o", "-"),
    "far_center":   ("#d6604d", "s", "-"),
    "far_L":        ("#f4a582", "^", "-"),
    "close_R":      ("#2166ac", "o", "--"),
    "close_center": ("#4393c3", "s", "--"),
    "close_L":      ("#92c5de", "^", "--"),
}
WINDOWS = (("ENL", "precue", "ENL (pre-cue)"), ("cue", "cue", "post-cue"),
           ("lick", "lick", "post-lick"))


def _fig_root():
    """Where the analysis figures/JSONs live, from the config -- NOT a literal.

    `E:/cue_lick` is the analysis box's path and would be wrong on the imaging box;
    tests/test_no_hardcoded_machine_paths.py fails the build for exactly this, and did.
    """
    return Path(PathResolver().root("figures_working"))


def _day(animal, mmdd):
    """Days from that animal's OWN lesion date. Negative = pre-stroke.

    `config.stroke_date` returns MMDD ('0817'), not YYYYMMDD -- taking it for the longer form
    silently yields an empty slice and an int() crash. Month*31 is a within-year ordering, not a
    calendar difference; it is monotone and that is all the x-axis needs, but do not read a gap of
    "31" as a month.
    """
    def ord_(s):
        return int(s[:2]) * 31 + int(s[2:])
    return ord_(mmdd) - ord_(str(config.stroke_date(animal)))


def _sessions(animal, phases=("pre", "post")):
    """(mmdd, day) for this animal's registered sessions, EXCLUDED ones dropped."""
    out = []
    for s in config.load_sessions():
        lab = s["label"]
        if not lab.startswith(animal):
            continue
        mmdd = lab.split("_")[1]
        ph = config.session_phase(animal, mmdd)
        if ph in phases:
            out.append((mmdd, _day(animal, mmdd)))
    return sorted(set(out), key=lambda t: t[1])


# ------------------------------------------------------------------ 1. behaviour
def _position_metrics(animal, mmdd):
    """{position: (hit_rate, ci_lo, ci_hi, n_engaged)} from the behaviour per-session CSV."""
    root = Path(PathResolver().root("behavior_out")) / "sessions" / animal / f"2026{mmdd}"
    if not root.exists():
        return {}
    files = sorted(root.glob("*position_metrics.csv"))
    if not files:
        return {}
    out = {}
    with files[-1].open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["pos_name"]] = (float(r["hit_rate"]), float(r["ci_lo"]), float(r["ci_hi"]),
                                      int(r["trials_engaged"]))
            except (ValueError, KeyError):
                continue
    return out


#: Sessions earlier than this many days before the lesion are collapsed into ONE baseline point.
#: The June block sits at day -70 and the whole post-stroke story inside +/-8, so plotting the true
#: axis spends 85% of the width on empty space and squeezes the result into a sliver.
BASELINE_BEFORE = -20
BASELINE_X = -16


def _wilson(hits, n, z=1.96):
    """Wilson 95% interval -- the same construction the per-session CSV uses, for pooled counts."""
    if not n:
        return (float("nan"),) * 3
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fig_behaviour(out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), sharey=True, squeeze=False)
    for k, an in enumerate(ANIMALS):
        ax = axes[0][k]
        for pos in POS:
            col, mk, ls = POS_STYLE[pos]
            pre_x, pre_y, pre_e = [], [], [[], []]
            post_x, post_y, post_e = [], [], [[], []]
            base_h = base_n = 0
            for mmdd, day in _sessions(an):
                m = _position_metrics(an, mmdd).get(pos)
                if not m or m[3] < 5:
                    continue
                if day < BASELINE_BEFORE:
                    base_h += round(m[0] * m[3]); base_n += m[3]
                    continue
                # CLAMP AT ZERO. A position at ceiling has hit_rate == ci_hi == 1.0 and the
                # subtraction lands on -1e-16, which matplotlib rejects outright rather than
                # rounding -- 51 cells across the cohort are at ceiling.
                tgt = (pre_x, pre_y, pre_e) if day < 0 else (post_x, post_y, post_e)
                tgt[0].append(day); tgt[1].append(m[0])
                tgt[2][0].append(max(0.0, m[0] - m[1])); tgt[2][1].append(max(0.0, m[2] - m[0]))
            if base_n:
                p, lo, hi = _wilson(base_h, base_n)
                ax.errorbar([BASELINE_X], [p], yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]],
                            color=col, marker=mk, ms=5.5, capsize=2, elinewidth=0.7, lw=0)
            # PRE AND POST ARE DRAWN AS SEPARATE SEGMENTS. Joining day -2 to day +1 draws a line
            # through the lesion and reads as a continuous decline that was measured; nothing was
            # recorded in between.
            for xs, ys, es in ((pre_x, pre_y, pre_e), (post_x, post_y, post_e)):
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ls=ls, ms=4.5, lw=1.4,
                                capsize=2, elinewidth=0.7,
                                label=pos if (xs is pre_x and k == 0) else None)
        ax.axvline(0, color="k", lw=1.6, ls=":")
        ax.text(0, 1.035, "lesion", ha="center", fontsize=8, color="k",
                transform=ax.get_xaxis_transform())
        ax.axvline((BASELINE_X + BASELINE_BEFORE) / 2 + 2, color="0.6", lw=1.0, ls=(0, (2, 3)))
        ax.text(BASELINE_X, -0.075, "June\nbaseline", ha="center", va="top", fontsize=7,
                color="0.35", transform=ax.get_xaxis_transform())
        ax.set_title(an, fontsize=12, fontweight="bold")
        ax.set_xlabel("days from lesion")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(BASELINE_X - 2.5, None)          # else the collapsed baseline sits on the spine
        if k == 0:
            ax.set_ylabel("hit rate (engaged trials)")
            ax.legend(fontsize=7, ncol=2, loc="lower left")
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Licking accuracy at each spout position, relative to the lesion. "
                 "Engaged trials only (the terminal quit period is excluded); bars are Wilson 95% "
                 "CIs. The two sessions after the laser that did not take are omitted.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out_dir) / "grant_1_behaviour_by_position.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_behaviour_collapsed(out_dir, jitter=0.11):
    """1b: EVERY pre-stroke day as one mean +/- SEM per position, then the post-stroke days.

    Priya, 2026-08-24. The dated version (panel 1) shows that baseline is flat and stays flat, which
    is worth showing once; after that the pre-stroke days are 11 points of the same number and the
    eye has to do the averaging. Here the whole baseline is ONE point per position and the deficit
    is read directly against it.

    SEM IS ACROSS SESSIONS, not across trials. The claim being made is "this position's hit rate on
    a typical day", so the day is the unit -- a trial-level interval would be several times tighter
    and would understate how much a position varies from session to session.

    Positions are offset horizontally because at baseline five of the six sit on top of each other
    near 1.0 and are otherwise unreadable.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.5), sharey=True, squeeze=False)
    for k, an in enumerate(ANIMALS):
        ax = axes[0][k]
        imp = _impaired(an)
        post_days = sorted({d for _m, d in _sessions(an, phases=("post",))})
        for pi, pos in enumerate(POS):
            col, mk, _ls = POS_STYLE[pos]
            off = (pi - (len(POS) - 1) / 2) * jitter
            pre_vals = [m[0] for mmdd, d in _sessions(an, phases=("pre",))
                        if (m := _position_metrics(an, mmdd).get(pos)) and m[3] >= 5]
            if pre_vals:
                ax.errorbar([-1 + off], [np.mean(pre_vals)],
                            yerr=[np.std(pre_vals, ddof=1) / np.sqrt(len(pre_vals))]
                            if len(pre_vals) > 1 else None,
                            color=col, marker=mk, ms=7, capsize=3, lw=0, elinewidth=1.2,
                            markeredgecolor="k", markeredgewidth=0.5,
                            # NO PER-POSITION ASTERISK IN THE LEGEND: the legend is drawn from
                            # the first animal only, so a mark meaning "impaired" there would be
                            # read as applying to all four. The per-panel title carries it.
                            label=pos if k == 0 else None)
            xs, ys = [], []
            for mmdd, d in _sessions(an, phases=("post",)):
                m = _position_metrics(an, mmdd).get(pos)
                if not m or m[3] < 5:
                    continue
                xs.append(d + off); ys.append(m[0])
            if xs:
                ax.errorbar(xs, ys, color=col, marker=mk, ms=4.5, lw=1.3, alpha=0.95)
        ax.axvline(-0.5, color="0.55", lw=1.0, ls=(0, (2, 3)))
        # NO "lesion" LABEL HERE -- it collided with the per-panel title, which carries the
        # impaired positions and is the more useful text. The dotted line plus the "ALL pre" tick
        # already say where the lesion is.
        ax.axvline(0.35, color="k", lw=1.6, ls=":")
        ax.set_xticks([-1] + post_days)
        ax.set_xticklabels(["ALL\npre"] + [str(d) for d in post_days], fontsize=8)
        ax.set_title(f"{an}\nimpaired: {', '.join(sorted(imp)) or 'none'}", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("days from lesion")
        ax.set_ylim(-0.02, 1.05)
        if k == 0:
            ax.set_ylabel("hit rate (engaged trials)")
            ax.legend(fontsize=7, ncol=2, loc="lower left")
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Licking accuracy per spout position: the WHOLE pre-stroke baseline as one point "
                 "(mean +/- SEM across sessions), then each day after the lesion.\n"
                 "Engaged trials only; positions offset horizontally so all six are visible. "
                 "Each panel title lists the positions that dropped below 50% on any post-stroke day.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out_dir) / "grant_1b_behaviour_pre_collapsed.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 2. pre-stroke decoding
def fig_prestroke_decoding(out_dir):
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    width = 0.26
    x = np.arange(len(ANIMALS))
    any_drawn = False
    for wi, (_disp, align, wname) in enumerate(WINDOWS):
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            print(f"  [grant] MISSING {f.name} -- run `joint_xsession --align {align}`", flush=True)
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        xs = x + (wi - (len(WINDOWS) - 1) / 2) * width
        for k, an in enumerate(ANIMALS):
            r = d.get(an)
            if not r:
                continue
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            vals = [v for lab, v in (r.get("per_session") or {}).items() if lab in pre]
            if not vals:
                continue
            m = float(np.mean(vals))
            # CI ACROSS HELD-OUT SESSIONS. The claim is that the code generalises to a session the
            # decoder never saw, so the session is the unit -- a trial-level CI would be far tighter
            # and would be answering a different question.
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.bar(xs[k], m, width * 0.92, color=f"C{wi}", edgecolor="k", linewidth=0.5,
                   label=wname if k == 0 else None, zorder=2)
            ax.errorbar(xs[k], m, yerr=1.96 * sem, color="k", capsize=3, lw=1.1, zorder=3)
            ax.plot(np.full(len(vals), xs[k]) + np.linspace(-0.05, 0.05, len(vals)), vals,
                    "o", ms=2.6, color="k", alpha=0.45, zorder=4)
            any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return None
    ax.axhline(1 / 6, color="k", ls=":", lw=1.2)
    ax.text(len(ANIMALS) - 0.4, 1 / 6 + 0.012, "chance (1/6)", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ANIMALS, fontsize=11)
    ax.set_ylabel("cross-session decoding accuracy")
    ax.set_ylim(0, 1.0)
    # LEGEND BELOW THE AXES. Bars run from 0 to ~0.95 in every animal, so there is no interior
    # corner it can sit in without covering data -- in-axes it landed on PS92's post-cue bar.
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Spout position decodes across sessions from a shared LocaNMF basis (pre-stroke)\n"
                 "Leave-one-session-out: the decoder never saw the session it is scored on.\n"
                 "Bar = mean over held-out sessions, error bar = 95% CI across sessions, "
                 "dots = individual sessions.", fontsize=9.5)
    fig.tight_layout()
    p = Path(out_dir) / "grant_2_prestroke_crossday_decoding.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_prestroke_decoding_cohort(out_dir):
    """2b: the cohort version of panel 2 -- one bar per window, all four animals.

    THE ANIMAL IS THE UNIT. Bar = mean of the four per-animal leave-one-session-out accuracies,
    error bar = SEM across ANIMALS (n=4), and each animal is drawn as a labelled point. Pooling all
    ~44 held-out sessions into one bar instead would give a far tighter interval that describes how
    much a SESSION varies, not how much an ANIMAL does -- and a cohort claim is a claim about
    animals. With n=4 the SEM is wide, which is honest: it is what four mice support.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    marks = ("o", "s", "^", "D")
    for wi, (_disp, align, wname) in enumerate(WINDOWS):
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        per_animal = []
        for an in ANIMALS:
            r = d.get(an)
            if not r:
                continue
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            vals = [v for lab, v in (r.get("per_session") or {}).items() if lab in pre]
            if vals:
                per_animal.append((an, float(np.mean(vals))))
        if not per_animal:
            continue
        ys = [v for _a, v in per_animal]
        m = float(np.mean(ys))
        sem = float(np.std(ys, ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
        ax.bar(wi, m, 0.62, color=f"C{wi}", edgecolor="k", linewidth=0.6, zorder=2, alpha=0.9)
        ax.errorbar(wi, m, yerr=sem, color="k", capsize=4, lw=1.3, zorder=3)
        for j, (an, v) in enumerate(per_animal):
            ax.plot(wi + (j - (len(per_animal) - 1) / 2) * 0.12, v, marks[j % len(marks)],
                    ms=6, color="k", mfc="white", zorder=4,
                    label=an if wi == 0 else None)
        ax.text(wi, m + sem + 0.03, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1 / 6, color="k", ls=":", lw=1.2)
    ax.text(len(WINDOWS) - 0.55, 1 / 6 + 0.015, "chance (1/6)", fontsize=8, ha="right")
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([w[2] for w in WINDOWS], fontsize=11)
    ax.set_ylabel("cross-session decoding accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Spout position decodes across sessions, all four animals (pre-stroke)\n"
                 "Leave-one-session-out in a shared LocaNMF basis. Bar = mean across animals, "
                 "error bar = SEM across animals (n=4),\npoints = individual animals.",
                 fontsize=9.5)
    fig.tight_layout()
    p = Path(out_dir) / "grant_2b_prestroke_crossday_cohort.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


#: The confusion matrices are stored in DISPLAY_ORDER -- the spatial layout of the spouts
#: (left-to-right, close row then far row), not the raw position codes. Labelling them in code order
#: would transpose the picture into nonsense while still looking like a plausible matrix.
CONF_LABELS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


def fig_confusion_prestroke(out_dir):
    """4: mean PRE-STROKE leave-one-session-out confusion, 2x2 animals, one file per window.

    Counts are SUMMED over the held-out pre-stroke sessions and then row-normalised, so each row is
    P(predicted | true) over the whole baseline -- not the mean of per-session rates, which would
    weight a 200-trial session the same as a 500-trial one.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.0), squeeze=False)
        drew = False
        for k, an in enumerate(ANIMALS):
            ax = axes[k // 2][k % 2]
            r = d.get(an)
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            mats = [np.array(m, float) for lab, m in ((r or {}).get("confusion") or {}).items()
                    if lab in pre]
            if not mats:
                ax.axis("off")
                continue
            C = np.sum(mats, axis=0)
            row = C.sum(1, keepdims=True)
            M = np.divide(C, row, out=np.zeros_like(C), where=row > 0)
            acc = float(np.trace(C) / C.sum()) if C.sum() else float("nan")
            im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
            for i in range(len(CONF_LABELS)):
                for j in range(len(CONF_LABELS)):
                    if M[i, j] >= 0.01:
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                                color="white" if M[i, j] < 0.6 else "black")
            ax.set_xticks(range(len(CONF_LABELS)))
            ax.set_xticklabels(CONF_LABELS, rotation=45, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(CONF_LABELS)))
            ax.set_yticklabels(CONF_LABELS, fontsize=7.5)
            ax.set_title(f"{an} — {acc:.2f} correct ({len(mats)} held-out sessions)",
                         fontsize=10, fontweight="bold")
            # X-LABEL ON THE BOTTOM ROW ONLY -- on the top row it lands on the row below's title.
            if k // 2 == 1:
                ax.set_xlabel("predicted")
            if k % 2 == 0:
                ax.set_ylabel("true")
            drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.035, pad=0.04, label="P(predicted | true)")
        fig.suptitle(f"Pre-stroke cross-session decoding — {wname} window\n"
                     "Frozen leave-one-session-out in the shared LocaNMF basis: every trial scored "
                     "by a decoder that never saw its session.\n"
                     "Counts summed over held-out sessions, then row-normalised. Chance = 0.17.",
                     fontsize=10)
        p = Path(out_dir) / f"grant_4_confusion_prestroke_{align}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)


def fig_confusion_pre_post(out_dir):
    """5: pre-stroke lick / pre-stroke NO-LICK control / post-stroke, one row per animal.

    THE MIDDLE PANEL IS WHY THIS IS THREE PANELS AND NOT TWO. Post-stroke the impaired positions are
    almost entirely no-lick trials, so an honest pre-vs-post pair on the ALL-trials arm compares
    pre-stroke LICK rows against post-stroke NON-LICK rows and confounds the lesion with the absence
    of a movement. `pre_nolick` is the matched control that already exists in `section_g`:
    PRE-stroke no-lick trials scored by a decoder trained on the OTHER pre-stroke sessions' engaged
    trials, so it differs from the post panel in PHASE ALONE. Read left-to-right: what the code
    looks like normally, what it looks like without a lick but without a lesion, and what it looks
    like after the lesion.

    Counts are reconstructed from the stored row-normalised matrices (matrix * n_per_true_position),
    summed across post-stroke sessions, and re-normalised. The `pre` and `pre_nolick` panels are the
    POOLED pre-stroke reference and are byte-identical in every session record -- summing them across
    sessions would multiply the same matrix by the session count and change nothing except to imply
    a sample size that does not exist.
    """
    sg = _fig_root() / "section_g.json"
    if not sg.exists():
        return None
    G = json.loads(sg.read_text(encoding="utf-8"))
    PANELS = (("pre", "PRE-stroke, LICK trials"),
              ("pre_nolick", "PRE-stroke, NO-LICK\n(matched control)"),
              ("post", "POST-stroke, ALL trials"))
    made = []
    for gkey, wname in (("pre-cue", "ENL (pre-cue)"), ("post-cue", "post-cue")):
        fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(11.0, 14.0), squeeze=False)
        drew = False
        for ri, an in enumerate(ANIMALS):
            sessions = sorted(k for k in G if k.startswith(an)
                              and config.session_phase(an, k.split("_")[-1]) == "post")
            if not sessions:
                for ci in range(3):
                    axes[ri][ci].axis("off")
                continue
            blocks = [((G[s].get("arms") or {}).get("all") or {}).get("confusion", {}).get(gkey)
                      for s in sessions]
            blocks = [b for b in blocks if b]
            if not blocks:
                for ci in range(3):
                    axes[ri][ci].axis("off")
                continue
            for ci, (key, ptitle) in enumerate(PANELS):
                ax = axes[ri][ci]
                use = blocks if key == "post" else blocks[:1]
                C = None
                for b in use:
                    d = b.get(key)
                    if not d:
                        continue
                    M = np.array(d["matrix"], float)
                    n = np.array(d["n_per_true_position"], float)
                    C = (np.nan_to_num(M) * n[:, None]) if C is None \
                        else C + np.nan_to_num(M) * n[:, None]
                if C is None:
                    ax.axis("off")
                    continue
                row = C.sum(1, keepdims=True)
                P = np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)
                acc = float(np.nansum(np.diag(C)) / C.sum()) if C.sum() else float("nan")
                im = ax.imshow(np.ma.masked_invalid(P), vmin=0, vmax=1, cmap="magma")
                for i in range(len(CONF_LABELS)):
                    if row[i, 0] == 0:
                        ax.text(2.5, i, "no trials", ha="center", va="center", fontsize=7,
                                color="firebrick", fontweight="bold")
                        continue
                    for j in range(len(CONF_LABELS)):
                        if P[i, j] >= 0.02:
                            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                    color="white" if P[i, j] < 0.6 else "black")
                ax.set_xticks(range(len(CONF_LABELS)))
                ax.set_xticklabels(CONF_LABELS if ri == len(ANIMALS) - 1 else [],
                                   rotation=45, ha="right", fontsize=6.5)
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_yticklabels(CONF_LABELS if ci == 0 else [], fontsize=6.5)
                ax.set_title(f"{an if ci == 0 else ''}  {ptitle}  ({acc:.2f})", fontsize=8.5,
                             fontweight="bold" if ci == 0 else "normal")
                drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.03, label="P(predicted | true)")
        fig.suptitle(f"The frozen pre-stroke decoder before and after the lesion — {wname} window\n"
                     "Rows = TRUE spout position, columns = predicted. MIDDLE PANEL IS THE MATCHED "
                     "CONTROL: pre-stroke NO-LICK trials scored by a decoder trained on the other "
                     "pre-stroke sessions,\nso it differs from the post panel in PHASE alone rather "
                     "than in phase and the absence of a movement together. "
                     "Post-stroke sessions pooled. Chance = 0.17.", fontsize=9.5)
        p = Path(out_dir) / f"grant_5_confusion_pre_post_{gkey.replace('-', '')}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)


# ------------------------------------------------------------------ 3a. coding retained
def _impaired(an, thresh=0.5, min_n=10):
    """Positions that DROPPED below `thresh` on any post-stroke session, from behaviour alone.

    THE WORST SESSION, not the pooled rate. Pooling across every post-stroke day averages a
    recovery away: PS95's far_R goes 0.00 on day 1 to 0.87 by day 2, which pools to 0.48-0.55 and
    reported that animal as having NO impaired position at all -- in the animal whose day-1 far_R
    collapse is the cleanest in the cohort. "Positions with a licking deficit" means positions that
    HAD one.
    """
    worst = {}
    for mmdd, _day_ in _sessions(an, phases=("post",)):
        for pos, (hr, _lo, _hi, n) in _position_metrics(an, mmdd).items():
            if n >= min_n:
                worst[pos] = min(worst.get(pos, 1.0), hr)
    return {p for p, v in worst.items() if v < thresh}


def fig_coding_retained(out_dir, meth="dom_orth"):
    """Two lines per panel, not six.

    THE SIX-LINE VERSION DID NOT WORK and the reason is structural, not cosmetic. It read the
    `poststroke_lick` class, so at an IMPAIRED position -- where the animal barely licks -- there is
    no cell to plot, and the positions the figure exists to describe were the ones missing from it.
    Twelve panels of six overlapping traces also had no legible message.

    So: the impaired positions are shown from MISS-WHILE-WORKING trials (the only trials they have)
    and the preserved positions from LICK trials, averaged within each group, SEM across positions.
    The two lines therefore come from different trial classes ON PURPOSE, which is stated on the
    figure -- an impaired position has no lick trials to average, and pretending otherwise is what
    produced the empty panels.
    """
    src = _fig_root() / "coding_direction.json"
    if not src.exists():
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(len(WINDOWS), len(ANIMALS), figsize=(16.0, 9.0),
                             sharey="row", squeeze=False)
    for ri, (disp, _align, wname) in enumerate(WINDOWS):
        for ci, an in enumerate(ANIMALS):
            ax = axes[ri][ci]
            ax.axhline(1.0, color="tab:green", ls=":", lw=1.4)
            ax.axhline(0.0, color="k", lw=0.8)
            res = (data.get(disp) or {}).get(an)
            imp = _impaired(an)
            if res and meth in res.get("methods", {}):
                by_cls = res["methods"][meth].get("cross_by_session", {})
                for group, cls, col, mk, lbl in (
                        (imp, "poststroke_miss_working", "#b2182b", "o",
                         "IMPAIRED positions (miss trials)"),
                        (set(POS) - imp, "poststroke_lick", "#2166ac", "s",
                         "positions still licked (lick trials)")):
                    cs = by_cls.get(cls, {})
                    xs, ys, es = [], [], []
                    for lab in sorted(cs):
                        vals = [(cs[lab].get(p) or {}).get(p) or {} for p in group]
                        vals = [c["mean"] for c in vals
                                if c.get("mean") is not None and (c.get("n") or 0) >= 10]
                        if not vals:
                            continue
                        xs.append(_day(an, lab.split("_")[-1]))
                        ys.append(float(np.mean(vals)))
                        es.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                                  if len(vals) > 1 else 0.0)
                    if xs:
                        order = np.argsort(xs)
                        ax.errorbar(np.array(xs)[order], np.array(ys)[order],
                                    yerr=np.array(es)[order], color=col, marker=mk, ms=5, lw=1.8,
                                    capsize=3, elinewidth=0.9,
                                    label=lbl if (ri == 0 and ci == 0) else None)
                        ax.set_xticks(sorted({int(v) for v in xs}))
            if ri == 0:
                ax.set_title(f"{an}\nimpaired: {', '.join(sorted(imp)) or 'none'}", fontsize=10,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{wname}\ncoding retained")
            if ri == len(WINDOWS) - 1:
                ax.set_xlabel("days from lesion")
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle("How much of each position's PRE-STROKE code survives, over days after the lesion.\n"
                 "1.0 (green) = that position's own pre-stroke signature; 0 = indistinguishable from "
                 "the other positions. Mean over positions in each group, error bars = SEM across "
                 "positions.\nThe two groups use DIFFERENT trial classes because an impaired "
                 "position has almost no lick trials to average.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = Path(out_dir) / "grant_3a_coding_retained.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 3b. frozen vs within
def _binom_ci(p, n):
    """Half-width of a normal-approx binomial 95% CI, 0 when n is 0."""
    return 1.96 * float(np.sqrt(max(p * (1 - p), 0) / n)) if n else 0.0


#: (window, section_g condition key, display name, which ARM it comes from).
#:
#: POST-LICK COMES FROM THE LICK-ONLY ARM, and that is not a workaround -- it is the only arm in
#: which the condition is defined. The ALL-trials arm includes trials with no detected lick and a
#: lick-aligned window cannot be built for a trial with no lick, so `poststroke_section_g` skips it
#: there (`if align == "lick" and arm_all: continue`) and computes it for lick-only, where every
#: trial has a lick by construction. It has been computed all along, in all 24 session records, with
#: permutation nulls and the pre-stroke band -- I asserted otherwise on 2026-08-24 after checking
#: `arms["all"]` alone and generalising from one arm to the analysis.
#:
#: THE COST, which is why the row is drawn with its own chance line: the lick-only arm scores each
#: session on ITS OWN preserved positions, so the class count and therefore chance differ between
#: sessions -- 4-way at 0.25 on one day and 6-way at 0.167 on another. Accuracies in that row are
#: NOT comparable across sessions or with the two rows above it.
FROZEN_WINDOWS = (("ENL", "pre-cue", "ENL (pre-cue)", "all"),
                  ("cue", "post-cue", "post-cue", "all"),
                  ("lick", "post-lick", "post-lick  [lick-only arm]", "lickonly"))


def fig_frozen_vs_within(out_dir):
    sg = _fig_root() / "section_g.json"
    if not sg.exists():
        return None
    G = json.loads(sg.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(len(FROZEN_WINDOWS), len(ANIMALS), figsize=(16.0, 10.2),
                             sharey="row", squeeze=False)
    for ri, (_disp, gkey, wname, armkey) in enumerate(FROZEN_WINDOWS):
        for ci, an in enumerate(ANIMALS):
            ax = axes[ri][ci]
            fx, fy, fe, wx, wy, we, band = [], [], [], [], [], [], None
            chance_x, chance_y = [], []
            for sess in sorted(k for k in G if k.startswith(an)):
                mmdd = sess.split("_")[-1]
                if config.session_phase(an, mmdd) != "post":
                    continue
                day = _day(an, mmdd)
                arm = (G[sess].get("arms") or {}).get(armkey) or {}
                if arm.get("chance"):
                    chance_x.append(day); chance_y.append(arm["chance"])
                cell = arm.get(gkey) or {}
                n = cell.get("n") or 0
                if cell.get("accuracy") is not None:
                    fx.append(day); fy.append(cell["accuracy"])
                    fe.append(_binom_ci(cell["accuracy"], n))
                # WITHIN-SESSION LIVES IN section_g TOO, under "<cond> within-session" -- but as a
                # POOLED block carrying a per_session LIST across every session, not a scalar for
                # this one. Reading it as a scalar silently produced nothing; reading it from
                # poststroke_grid.json (the other obvious source) gives only day 1 and 2, which is
                # one or two points per animal and no trajectory at all.
                wblk = arm.get(f"{gkey} within-session") or {}
                if wblk and band is None:
                    band = wblk.get("within_pre_band")
                # EACH SESSION'S BLOCK CARRIES ONLY ITS OWN POST ROW -- the rest of `per_session` is
                # that animal's pre-stroke sessions, which is what the band is built from. Taking
                # the list from the first session and stopping (the obvious read) yields exactly one
                # green point per animal, which is what the first version of this figure showed.
                for row in wblk.get("per_session", []):
                    if row.get("post") and row.get("label") == sess \
                            and row.get("within_accuracy") is not None:
                        wx.append(day); wy.append(row["within_accuracy"])
                        we.append(_binom_ci(row["within_accuracy"], row.get("n") or n))
            if wx:
                order = np.argsort(wx)
                wx = list(np.array(wx)[order]); wy = list(np.array(wy)[order])
                we = list(np.array(we)[order])
            if band:
                ax.axhspan(band["min"], band["max"], color="tab:blue", alpha=0.15, zorder=1,
                           label="pre-stroke range" if (ri == 0 and ci == 0) else None)
            if fx:
                ax.errorbar(fx, fy, yerr=fe, color="tab:red", marker="o", ms=5, lw=1.6, capsize=3,
                            label="FROZEN pre-stroke decoder" if (ri == 0 and ci == 0) else None)
            if wx:
                ax.errorbar(wx, wy, yerr=we, color="tab:green", marker="s", ms=5, lw=1.6, capsize=3,
                            ls="--",
                            label="trained on that session" if (ri == 0 and ci == 0) else None)
            # CHANCE IS PER SESSION IN THE LICK-ONLY ARM. One flat 1/6 line would be wrong on
            # every 4-position session, and drawing it anyway is how a 4-way 0.5 gets read as
            # twice chance when it is exactly twice a DIFFERENT chance.
            if armkey == "lickonly" and chance_x:
                o = np.argsort(chance_x)
                ax.step(np.array(chance_x)[o], np.array(chance_y)[o], where="mid", color="k",
                        ls=":", lw=1.1)
            else:
                ax.axhline(1 / 6, color="k", ls=":", lw=1.0)
            ax.set_ylim(0, 1.02)
            if fx or wx:
                ax.set_xticks(sorted({int(v) for v in list(fx) + list(wx)}))
            if ri == 0:
                ax.set_title(an, fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{wname}\naccuracy")
            if ri == len(FROZEN_WINDOWS) - 1:
                ax.set_xlabel("days from lesion")
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Does the OLD code still read out, and is position information still there?\n"
                 "RED = frozen pre-stroke decoder.  GREEN = decoder trained on that session.  "
                 "Band = pre-stroke range.  Bars = binomial 95% CIs.\n"
                 "Top two rows: all trials, 6 positions, chance 1/6. Bottom row: LICK-ONLY arm "
                 "(a trial with no lick has no lick-aligned window), so chance is per session "
                 "(dotted step) and those panels are NOT comparable across sessions.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = Path(out_dir) / "grant_3b_frozen_vs_within.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--only", nargs="+", default=None,
                    choices=("1", "1b", "2", "2b", "3a", "3b", "4", "5"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1", "1b", "2", "2b", "3a", "3b", "4", "5"))
    jobs = (("1", fig_behaviour), ("1b", fig_behaviour_collapsed),
            ("2", fig_prestroke_decoding), ("2b", fig_prestroke_decoding_cohort),
            ("3a", fig_coding_retained), ("3b", fig_frozen_vs_within),
            ("4", fig_confusion_prestroke), ("5", fig_confusion_pre_post))
    for key, fn in jobs:
        if key not in want:
            continue
        try:
            p = fn(out)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {key}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
            continue
        print(f"  {'wrote ' + str(p) if p else f'{key}: no data'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

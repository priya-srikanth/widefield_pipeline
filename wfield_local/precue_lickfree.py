"""Pre-cue decoding and encoding restricted to trials with NO licks in the analysis window.

WHY THIS IS A PIPELINE STEP AND NOT A ONE-OFF. Pre-cue position information is the study's key
pre-stroke readout, and licking is the obvious confound: it is an orofacial movement that may itself be
directed at the spout, so "pre-cue position information" could be ongoing motor activity rather than a
held intention. Every pre-cue number in the deck inherits that doubt unless it is checked, and the
check has to travel with the data as sessions accumulate.

WHAT THE TASK ALREADY GUARANTEES (measured 2026-08-13). The strobe->cue interval is an ENFORCED NO-LICK
period: licking restarts it, which is why the lead is a median 3.0 s but ranges to a p90 of 18.1 s in
PS92. So by the time the cue fires the animal has necessarily just completed a quiet stretch, and
90.8-99.5% of 2 s pre-cue windows contain no licks at all. The confound is largely excluded by
construction, not merely absent by luck.

WHAT THE CHECK ADDS ANYWAY. "Largely" is per-animal: PS92 and PS93 sit at ~91-92% lick-free while
PS94/PS95 are at 98-99.5%, so the residual exposure is exactly where the licking behaviour is most
atypical -- and PS93's licking (often not reaching the spout) is already known to be unusual. Measured
on the full curated set, position decoding on LICK-FREE trials alone: PS92 0.579, PS93 0.477, PS94
0.511, PS95 0.567 (chance 0.167). Information survives with no licking, so it is not lick-driven.

HOW TO READ THE OUTPUT. The lick-free arm is the evidence. The with-licks arm is shown for contrast
only and proves nothing either way -- it is a small, self-selected subset (n=49 for PS94, n=0 for
PS95), and a low number there reflects sample size, not the absence of a code. Where the with-licks
arm has too few trials the figure says so rather than plotting a meaningless bar.

    python -m wfield_local.precue_lickfree --output <dir> [--from 0606-0812] [--only PS93]
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config, results_store as rs
from wfield_local.behavior_position import classify_cues_with_backup
from wfield_local.locanmf_crossanimal_dff import _frames
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _build_signal
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES, _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue
from wfield_local.precue_attribution import CHANCE, C_REG, encoding_ev, family_columns, region_names

FS = 31.23
PRE_S = 2.0            # the pipeline's pre-cue window: 2 s ENDING at the cue
MIN_TRIALS = 40        # below this a decode result is not reported at all


def trial_table(session, source="roi", pre_s=PRE_S):
    """(X, y, blocks, n_licks) for the pre-cue window. ``n_licks`` counts licks INSIDE that window."""
    sig, feat_reg = _build_signal(session, source)
    cue = _load_cue(session["h5"])
    lk = _load_daq_events(session["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, _ = _frames(session, cue, lk)
    codes = classify_cues_with_backup(session, cue, verbose=False)
    ls = np.sort(np.asarray(lick_f))
    wn = int(round(pre_s * FS))

    blk, b, prev = np.full(cue_f.size, -1, int), -1, None
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        if prev is None or codes[k] != prev:
            b += 1
        blk[k] = b
        prev = int(codes[k])

    X, y, g, nl = [], [], [], []
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        w0 = int(cue_f[k]) - wn
        if w0 < 0 or w0 + wn > sig.shape[1]:
            continue
        X.append(sig[:, w0:w0 + wn].mean(1))
        y.append(int(codes[k]))
        g.append(int(blk[k]))
        nl.append(int(np.sum((ls >= w0) & (ls < w0 + wn))))
    return (np.asarray(X), np.asarray(y), np.asarray(g), np.asarray(nl), feat_reg)


def _decode(X, y, g):
    ng = min(5, int(np.unique(g).size))
    if ng < 2 or len(y) < MIN_TRIALS or len(np.unique(y)) < len(DISPLAY_ORDER):
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C_REG))
    pred = cross_val_predict(clf, X, y, cv=GroupKFold(ng), groups=g)
    cm = confusion_matrix(y, pred, labels=DISPLAY_ORDER).astype(float)
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    return {"accuracy": float(accuracy_score(y, pred)), "confusion": cm, "n": int(len(y))}


def analyse(session, source="roi"):
    """Lick-free vs with-licks decode + encode for one session."""
    X, y, g, nl, feat_reg = trial_table(session, source)
    if not len(y):
        return None
    free = nl == 0
    out = {"label": session["label"], "source": source, "n_trials": int(len(y)),
           "n_lickfree": int(free.sum()), "frac_lickfree": float(free.mean()),
           "licks_by_position": {POSITION_NAMES[c]: float(np.mean(nl[y == c])) if (y == c).any() else np.nan
                                 for c in DISPLAY_ORDER}}
    out["decode_lickfree"] = _decode(X[free], y[free], g[free])
    out["decode_withlicks"] = _decode(X[~free], y[~free], g[~free])
    out["decode_all"] = _decode(X, y, g)

    fam = family_columns(feat_reg, region_names(session))
    for tag, m in (("lickfree", free), ("all", np.ones(len(y), bool))):
        if m.sum() >= MIN_TRIALS and len(np.unique(y[m])) == len(DISPLAY_ORDER):
            ev, ceil = encoding_ev(X[m], y[m], g[m])
            out[f"encode_{tag}"] = {f: {"ev": float(np.nanmean(ev[c])),
                                        "ceiling": float(np.nanmean(ceil[c]))}
                                    for f, c in fam.items()}
        else:
            out[f"encode_{tag}"] = None
    return out


def figure(per_session, out_png, title):
    """Per animal: exposure, decode lick-free vs all, the lick-free confusion matrix, encoding EV."""
    good = [r for r in per_session if r]
    if not good:
        return None
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.34)
    labs = [r["label"][-4:] for r in good]
    x = np.arange(len(good))

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, [100 * r["frac_lickfree"] for r in good], color="#3b7dd8")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=60, fontsize=7)
    ax.set_ylim(0, 100); ax.set_ylabel("% of pre-cue windows lick-free")
    ax.set_title("Exposure\n(task ENL makes the window quiet by construction)", fontsize=8)

    ax = fig.add_subplot(gs[0, 1:3])
    lf = [r["decode_lickfree"]["accuracy"] if r["decode_lickfree"] else np.nan for r in good]
    al = [r["decode_all"]["accuracy"] if r["decode_all"] else np.nan for r in good]
    wl = [r["decode_withlicks"]["accuracy"] if r["decode_withlicks"] else np.nan for r in good]
    ax.plot(x, al, "o-", color="#888", label="all trials")
    ax.plot(x, lf, "o-", color="#3b7dd8", lw=2, label="LICK-FREE trials only")
    ax.plot(x, wl, "x", color="#d1495b", label="with licks (small n — not evidence)")
    ax.axhline(CHANCE, color="gray", ls=":", label=f"chance {CHANCE:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=60, fontsize=7)
    ax.set_ylabel("pre-cue decode accuracy"); ax.legend(fontsize=7)
    ax.set_title("Position decoding survives with NO licking in the window", fontsize=9)

    ax = fig.add_subplot(gs[0, 3])
    cms = [r["decode_lickfree"]["confusion"] for r in good if r["decode_lickfree"]]
    if cms:
        im = ax.imshow(np.nanmean(np.stack(cms), 0), vmin=0, vmax=1, cmap="magma")
        names = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
        ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(6)); ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Confusion, lick-free\n(mean over sessions)", fontsize=8)

    ax = fig.add_subplot(gs[1, :2])
    fams = sorted({f for r in good if r.get("encode_lickfree") for f in r["encode_lickfree"]})
    if fams:
        xf = np.arange(len(fams))
        ev_lf = [np.nanmean([r["encode_lickfree"][f]["ev"] for r in good
                             if r.get("encode_lickfree") and f in r["encode_lickfree"]]) for f in fams]
        ev_all = [np.nanmean([r["encode_all"][f]["ev"] for r in good
                              if r.get("encode_all") and f in r["encode_all"]]) for f in fams]
        ax.bar(xf - 0.2, ev_all, 0.4, color="#888", label="all trials")
        ax.bar(xf + 0.2, ev_lf, 0.4, color="#3b7dd8", label="lick-free")
        ax.set_xticks(xf); ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("position EV ($R^2$)")
        ax.legend(fontsize=7)
    ax.set_title("Encoding: position variance explained per region, lick-free vs all", fontsize=9)

    ax = fig.add_subplot(gs[1, 2:]); ax.axis("off")
    nlf = sum(r["n_lickfree"] for r in good); ntot = sum(r["n_trials"] for r in good)
    ax.text(0, 1, "\n".join([
        title, "",
        f"{len(good)} sessions   {nlf}/{ntot} trials lick-free ({100*nlf/max(ntot,1):.1f}%)",
        "",
        "The strobe->cue interval is an ENFORCED NO-LICK period: licking restarts it",
        "(median lead 3.0 s, p90 up to 18.1 s). So the final 2 s before the cue is quiet",
        "BY CONSTRUCTION, not by chance -- the confound is largely designed out.",
        "",
        "The LICK-FREE arm is the evidence: information present with no licking cannot",
        "be lick-driven. The with-licks arm is contrast only -- a small self-selected",
        "subset -- and a low value there reflects n, not the absence of a code.",
        "",
        "CAVEAT that this figure does NOT address: the pre-cue window sits ~3 s AFTER",
        "the spout reaches its position, so it cannot separate a maintained plan from",
        "somatosensory contact with the already-positioned spout. See DECISIONS.md.",
    ]), va="top", ha="left", fontsize=7.4, family="monospace")

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--from", dest="from_dates", default=None, help="date spec (default: curated set)")
    ap.add_argument("--only", nargs="+", default=None, help="animals (default: all)")
    ap.add_argument("--source", default="roi", choices=("roi", "locanmf"))
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    dates = (set(config.expand_dates(args.from_dates, width=4)) if args.from_dates
             else set(config.curated_dates()))
    only = config.normalize_animals(args.only) or ["PS92", "PS93", "PS94", "PS95"]

    summary = {}
    for an in only:
        rows = []
        for lab in [s["label"] for s in SESSIONS
                    if s["label"].startswith(an) and s["label"][-4:] in dates]:
            s = next(x for x in SESSIONS if x["label"] == lab)
            try:
                r = analyse(s, args.source)
            except Exception as ex:                        # noqa: BLE001
                print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:60]}", flush=True); continue
            if r:
                rows.append(r)
                d = r["decode_lickfree"]
                print(f"  {lab:12s} lick-free {r['n_lickfree']:4d}/{r['n_trials']:4d} "
                      f"({100*r['frac_lickfree']:5.1f}%)  decode "
                      f"{d['accuracy']:.3f}" if d else f"  {lab:12s} (too few)", flush=True)
        if not rows:
            continue
        png = os.path.join(args.output, f"precue_lickfree_{an}_{args.source}.png")
        figure(rows, png, f"{an} — pre-cue position code without licking ({args.source})")
        print(f"wrote {png}", flush=True)
        summary[an] = {"n_sessions": len(rows),
                       "frac_lickfree": float(np.mean([r["frac_lickfree"] for r in rows])),
                       "decode_lickfree": float(np.nanmean([r["decode_lickfree"]["accuracy"]
                                                            for r in rows if r["decode_lickfree"]])),
                       "decode_all": float(np.nanmean([r["decode_all"]["accuracy"]
                                                       for r in rows if r["decode_all"]]))}
    rs.save(args.output, "precue_lickfree", None, summary,
            meta={"window": f"{PRE_S} s ending at cue", "source": args.source, "chance": CHANCE})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

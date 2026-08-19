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

WHAT THE CHECK ADDS ANYWAY. "Largely" is per-SESSION, not per-animal: PS93 sits at 98% lick-free in
June but falls to 76% on 8/9. A per-animal average would hide exactly the sessions where the licking
behaviour is most atypical -- and PS93's licking (often not reaching the spout) is already unusual.

THE WINDOW IS SEARCHED, NOT FIXED (Priya, 2026-08-13). Requiring the fixed 2 s ending at the cue to be
clean discards a whole trial for one lick anywhere in it -- including a lick 200 ms before the cue,
when 2 s of clean data sits just earlier in the same no-lick period. Instead ``lickfree_window`` takes
the lick-free GAPS in [strobe, cue] and uses the last 2 s of the LATEST gap wide enough to hold the
window: still 2 s, still pre-cue, still lick-free, but recovered rather than thrown away. Trials whose
fixed window is already clean keep it, so the common case stays exactly cue-aligned.

The search is BOUNDED AT THE SPOUT STROBE. Earlier than that the position for this trial does not yet
exist, and because the task avoids recent repeats, prior-trial activity carries real information about
the upcoming position -- a window straying before the strobe would smuggle that in and look like a
pre-cue code. ``window_offset_s_*`` reports how far each recovered window sits from the cue, because
recovered trials are no longer perfectly cue-aligned and that trade-off should be visible, not hidden.

HOW TO READ THE OUTPUT. The lick-free arm is the evidence. The with-licks arm -- now only trials where
NO clean 2 s window exists anywhere in the interval -- is contrast only and proves nothing either way:
it is a small, self-selected subset, so a low value there reflects sample size, not the absence of a
code.

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
from wfield_local.locanmf_position_decoder import _bins_for, _build_signal, _window_feature

#: Sub-bins for the pre-cue window. MUST match the production decoder: this module is the
#: CONTROL for the pre-cue readout, and a control built on different features cannot be
#: compared with the thing it controls -- a lower lick-free number would be ambiguous between
#: 'licking mattered' and '66 features instead of 264'. It previously used a single 2 s mean
#: while Section A used 4 x 0.5 s.
_BINS = _bins_for(SimpleNamespace(align='precue', bins=None))
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES, _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue
from wfield_local.precue_attribution import CHANCE, C_REG, encoding_ev, family_columns, region_names

FS = 31.23
PRE_S = 2.0            # the pipeline's pre-cue window: 2 s ENDING at the cue
MIN_TRIALS = 40        # below this a decode result is not reported at all
ENL_LICKFREE_WARN = 0.85   # below this, say so: the ENL assumption is the pre-cue readout's floor


#: The rule that gates the HEADLINE pre-cue number, so it lives beside the
#: production decoder rather than in this module, which used to be its only
#: consumer. Re-exported here because callers and DECISIONS.md name it here.
from wfield_local.locanmf_position_decoder import lickfree_window


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

    # spout arrival, to bound the search (see lickfree_window)
    cs = np.asarray(cue["cue_samples"]); ss = np.asarray(cue["strobe_samples"])
    sr = float(cue["sample_rate_hz"])
    j = np.searchsorted(ss, cs, side="right") - 1
    lead_s = np.where(j >= 0, (cs - ss[np.clip(j, 0, len(ss) - 1)]) / sr, np.nan)
    strobe_f = cue_f - lead_s * FS

    X, y, g, nl, off = [], [], [], [], []
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        fixed = int(cue_f[k]) - wn                       # the fixed window ending at the cue
        if fixed < 0 or fixed + wn > sig.shape[1]:
            continue
        n_fixed = int(np.sum((ls >= fixed) & (ls < fixed + wn)))
        if n_fixed == 0:
            w0, nlicks = fixed, 0                        # already clean; keep it cue-aligned
        else:
            w0 = lickfree_window(cue_f[k], strobe_f[k], ls, wn)
            if w0 is None or w0 < 0 or w0 + wn > sig.shape[1]:
                w0, nlicks = fixed, n_fixed              # no clean window exists -> keep, flagged dirty
            else:
                nlicks = 0
        X.append(_window_feature(sig, int(w0), wn, _BINS, 0.0))
        y.append(int(codes[k]))
        g.append(int(blk[k]))
        nl.append(nlicks)
        off.append((int(cue_f[k]) - (w0 + wn)) / FS)     # seconds between window end and the cue
    if _BINS > 1:
        feat_reg = np.tile(feat_reg, _BINS)
    return (np.asarray(X), np.asarray(y), np.asarray(g), np.asarray(nl), feat_reg,
            np.asarray(off, dtype=float))


ROLL_WIN_S, ROLL_STEP_S = 0.5, 0.25
ROLL_FROM_S, ROLL_TO_S = -3.5, 2.5      # deep ENL through post-cue; matches fig_rolling_cue


def rolling_lickfree(session, source="roi", win_s=ROLL_WIN_S, step_s=ROLL_STEP_S):
    """Cue-aligned sliding-window decoding, with licking excluded PER TIME BIN.

    Why this supersedes the single-window version (Priya, 2026-08-13). A 2 s mean forces one window
    choice and makes trial inclusion binary: a lick anywhere in the window costs the whole trial, and
    shortening the window to rescue those trials costs noise-averaging on every OTHER trial. Measured
    recovery for a clean window of length L in the worst sessions (PS93 8/9, PS92 8/9): 2.0 s -> ~90%,
    1.8 s -> ~95%, 1.5 s -> ~99%, 1.2 s -> 100%.

    A 0.5 s rolling window dissolves that trade rather than managing it. Clean 0.5 s gaps are
    abundant, so per-bin exclusion drops almost nothing at any timepoint, and the output is a TIME
    COURSE spanning ENL -> post-cue instead of a single scalar. Exclusion is per (trial, bin), so a
    trial contributes wherever it is quiet and is dropped only from the bins it contaminates.

    Returns offsets (s, window CENTRE relative to cue), accuracy with and without lick filtering, and
    the surviving trial count per bin -- the count is reported because it is the honest denominator: a
    bin decoded from few trials is not comparable to one decoded from all of them.
    """
    sig, _ = _build_signal(session, source)
    cue = _load_cue(session["h5"])
    lk = _load_daq_events(session["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, _ = _frames(session, cue, lk)
    codes = classify_cues_with_backup(session, cue, verbose=False)
    ls = np.sort(np.asarray(lick_f))
    wn = int(round(win_s * FS))

    blk, b, prev = np.full(cue_f.size, -1, int), -1, None
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        if prev is None or codes[k] != prev:
            b += 1
        blk[k] = b
        prev = int(codes[k])

    offs = np.arange(ROLL_FROM_S, ROLL_TO_S + 1e-9, step_s)
    out = {"offsets_s": (offs + win_s / 2).tolist(), "win_s": win_s,
           "acc_all": [], "acc_lickfree": [], "n_lickfree": [], "n_all": []}
    for off in offs:
        Xa, ya, ga, clean = [], [], [], []
        for k in range(cue_f.size):
            if codes[k] < 0:
                continue
            w0 = int(round(cue_f[k] + off * FS))
            if w0 < 0 or w0 + wn > sig.shape[1]:
                continue
            Xa.append(sig[:, w0:w0 + wn].mean(1))
            ya.append(int(codes[k]))
            ga.append(int(blk[k]))
            clean.append(not ((ls >= w0) & (ls < w0 + wn)).any())
        Xa, ya, ga = np.asarray(Xa), np.asarray(ya), np.asarray(ga)
        clean = np.asarray(clean, bool)
        ra = _decode(Xa, ya, ga) if len(ya) else None
        rf = _decode(Xa[clean], ya[clean], ga[clean]) if clean.sum() else None
        out["acc_all"].append(ra["accuracy"] if ra else np.nan)
        out["acc_lickfree"].append(rf["accuracy"] if rf else np.nan)
        out["n_all"].append(int(len(ya)))
        out["n_lickfree"].append(int(clean.sum()))
    return out


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
    X, y, g, nl, feat_reg, off = trial_table(session, source)
    if not len(y):
        return None
    free = nl == 0
    frac = float(free.mean())
    # THE ENL LIVES IN FIRMWARE, NOT IN gui_config.json, so if it were shortened at the rig no config
    # this pipeline reads would change -- the pre-cue window would quietly start sampling licking and
    # every pre-cue number would keep being reported as motor-independent. Measured 2026-08-17 the
    # boundary is razor-sharp (licks within 2.0 s of a cue: 0.00-0.19% of all licks; within 3.0 s:
    # 1.7-10.7%), so the 2 s window is the MAXIMAL safe one and a real drop here means the task
    # changed or the animal's licking did. Warn, do not fail: PS93 legitimately reaches 76% on 8/9
    # through unusual licking alone, and a guard that fires on real behaviour gets ignored.
    if frac < ENL_LICKFREE_WARN:
        print(f"  !! {session['label']}: only {100*frac:.1f}% of {PRE_S}s pre-cue windows are "
              f"lick-free (expected >{100*ENL_LICKFREE_WARN:.0f}% under a {PRE_S}s ENL). Either this "
              f"animal's licking is atypical or the ENL changed -- the pre-cue readout's "
              f"motor-independence rests on this.", flush=True)
    out = {"label": session["label"], "source": source, "n_trials": int(len(y)),
           "n_lickfree": int(free.sum()), "frac_lickfree": frac,
           "enl_assumption_ok": bool(frac >= ENL_LICKFREE_WARN),
           "window_offset_s_median": float(np.nanmedian(off[nl == 0])) if (nl == 0).any() else np.nan,
           "window_offset_s_p90": float(np.nanpercentile(off[nl == 0], 90)) if (nl == 0).any() else np.nan,
           "n_window_moved": int(np.sum((nl == 0) & (off > 1e-6))),
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
            meta={"window": f"{PRE_S} s, SEARCHED: the latest lick-free {PRE_S} s in [strobe, cue] "
                            f"(the window ending at the cue when that is already clean)",
                  "source": args.source, "chance": CHANCE})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

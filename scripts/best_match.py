"""WHICH PRE-STROKE POSITION DOES POST-STROKE ACTIVITY LOOK LIKE? Per animal, window, class, session.

    PYTHONPATH=$(pwd) python scripts/best_match.py [ENL cue lick]

Priya, 2026-08-24: "for far_R trials in an animal that cannot lick far_R, does ENL activity look most
similar to pre-stroke ENL of far_R trials? vs pre-stroke far_L trials? What are the best matches, and
how does that change with miss-while-working vs stopped, and over recovery sessions?"

THE DATA ALREADY ANSWERS THIS. `coding_direction.json` stores a cross matrix per class: for trials of
each TRUE position, the projection onto EVERY position's pre-stroke direction, pole-normalised so
1.0 = that position's own pre-stroke lick signature. The best match is the argmax across columns.

TWO READINGS, BOTH REPORTED, BECAUSE THE RAW ARGMAX ALONE IS MISLEADING:

  RAW      argmax of the row. The literal answer to "what does it look most like" -- but positions
           are intrinsically similar before any lesion (pre-stroke far_center already scores ~0.76 on
           the far_R direction), so the raw winner is often just the neighbour, with no lesion in it.
  SHIFTED  the same row MINUS the PRE-STROKE LICK row for that same true position. This is what
           CHANGED, i.e. what the trials moved TOWARD, and it is the remapping question. A column
           that is high raw but flat in the shift was always that similar.

Read them together: a raw winner that is also the shift winner is a genuine move. A raw winner with a
NEGATIVE shift means the trials still resemble that position most, but LESS than they used to.

CAVEATS THAT LIMIT EVERY NUMBER HERE
  * Cells are gated on n: a column mean over a handful of trials is noise (`low_n`).
  * The ENL window is lick-free by construction, so nothing in it is a movement difference. The CUE
    window is not: a post-stroke miss trial contains no lick where its pre-stroke counterpart did, so
    a cue-window "match" partly reflects the presence or absence of a movement.
  * MISS and STOPPED are defined by spout CONTACT, not by attempting -- an off-target lick is scored
    as a miss. The pre-stroke reference has the same ambiguity.
  * These are dom_orth directions (engagement axis projected out). Audited 2026-08-24: that
    projection moves the estimate toward the covariance-aware reference in ENL for 4/4 animals, but
    AWAY for PS92 in the cue window -- so PS92's cue rows are the least trustworthy in this table.
"""
import json
import sys
from pathlib import Path

POS = ["far_R", "far_center", "far_L", "close_R", "close_center", "close_L"]
CLASSES = ("poststroke_lick", "poststroke_miss_working", "poststroke_stopped")
SRC = Path("E:/cue_lick/coding_direction.json")
SECTION_G = Path("E:/cue_lick/section_g.json")
METH = "dom_orth"
MIN_N = 20


def response_rates():
    """Post-stroke response rate per animal per position, from the behaviour counts."""
    if not SECTION_G.exists():
        return {}
    rec = json.loads(SECTION_G.read_text(encoding="utf-8"))
    tot = {}
    for sess, r in rec.items():
        an = sess.split("_")[0]
        c = (r.get("counts") or {}).get("post") or {}
        for p in POS:
            e = (c.get("engaged") or {}).get(p, 0)
            u = (c.get("undetected") or {}).get(p, 0)
            a = tot.setdefault(an, {}).setdefault(p, [0, 0])
            a[0] += e
            a[1] += u
    return {an: {p: (e / (e + u) if (e + u) else float("nan")) for p, (e, u) in d.items()}
            for an, d in tot.items()}


def row(cm, cls, true_pos):
    """{scored position: mean} for one true position, dropping thin cells."""
    r = ((cm.get(cls) or {}).get(true_pos)) or {}
    return {p: c["mean"] for p, c in r.items()
            if c and c.get("mean") is not None and (c.get("n") or 0) >= MIN_N}


def n_of(cm, cls, true_pos):
    r = ((cm.get(cls) or {}).get(true_pos)) or {}
    own = r.get(true_pos) or {}
    return int(own.get("n") or 0)


def best(d):
    return max(d, key=d.get) if d else None


def fmt(p, d):
    return f"{p}={d[p]:+.2f}" if p in d else "--"


def main(windows):
    rates = response_rates()
    data = json.loads(SRC.read_text(encoding="utf-8"))
    for window in windows:
        if window not in data:
            continue
        print(f"\n{'=' * 108}\n### {window} WINDOW\n")
        for an in ("PS92", "PS93", "PS94", "PS95"):
            res = (data[window] or {}).get(an)
            if not res or METH not in res.get("methods", {}):
                continue
            cm = res["methods"][METH]["cross_matrix"]
            rr = rates.get(an, {})
            impaired = sorted([p for p in POS if rr.get(p, 1) < 0.5], key=lambda p: rr.get(p, 1))
            print(f"{an}   post-stroke response rate: "
                  + "  ".join(f"{p}={rr[p]:.2f}" for p in POS if p in rr))
            print(f"   IMPAIRED (<0.50): {impaired or 'none'}")
            for cls in CLASSES:
                for tp in (impaired or POS[:1]):
                    r = row(cm, cls, tp)
                    pre = row(cm, "prestroke_lick", tp)
                    if not r:
                        print(f"   {cls:<24} {tp:<13} no cell with n>={MIN_N}")
                        continue
                    shift = {p: r[p] - pre[p] for p in r if p in pre}
                    braw, bsh = best(r), best(shift)
                    print(f"   {cls:<24} {tp:<13} n={n_of(cm, cls, tp):<5} "
                          f"own={fmt(tp, r):<18} RAW best {fmt(braw, r):<18} "
                          f"MOVED-toward {bsh}={shift[bsh]:+.2f}" if bsh else "")
            print()


def by_session(windows):
    """The RECOVERY view: best match per post-stroke session, impaired positions, miss class.

    Priya, 2026-08-24: on far_R miss trials she often sees incomplete LEFTWARD or CENTRAL licks
    after the cue. The two windows separate the two readings of that -- ENL precedes any movement,
    so a far_R match there is the PLAN; the cue window contains the attempted movement, so a far_L
    match there is cortex following the EXECUTED direction. Printed side by side per session.
    """
    rates = response_rates()
    data = json.loads(SRC.read_text(encoding="utf-8"))
    for an in ("PS92", "PS93", "PS94", "PS95"):
        rr = rates.get(an, {})
        impaired = sorted([p for p in POS if rr.get(p, 1) < 0.5], key=lambda p: rr.get(p, 1))
        if not impaired:
            continue
        print(f"\n{'=' * 100}\n### {an}   impaired: {impaired}\n")
        for tp in impaired:
            for cls in ("poststroke_miss_working", "poststroke_lick"):
                rows = []
                for window in windows:
                    res = (data.get(window) or {}).get(an)
                    if not res:
                        continue
                    cs = res["methods"][METH].get("cross_by_session", {}).get(cls, {})
                    for sess in sorted(cs):
                        cells = cs[sess].get(tp) or {}
                        r = {p: c["mean"] for p, c in cells.items()
                             if c and c.get("mean") is not None and (c.get("n") or 0) >= MIN_N}
                        if not r:
                            continue
                        b = best(r)
                        rows.append((sess.split("_")[-1], window,
                                     (cells.get(tp) or {}).get("n") or 0,
                                     r.get(tp), b, r[b]))
                if not rows:
                    continue
                print(f"  {tp}  [{cls}]")
                for date, window, n, own, b, bv in sorted(rows):
                    mark = "  <-- own" if b == tp else ""
                    ownf = "--" if own is None else f"{own:+.2f}"
                    print(f"     {date}  {window:<5} n={n:<5} own={ownf:<7} best={b} ({bv:+.2f})"
                          f"{mark}")
                print()


def preserved(window="lick"):
    """Q2: at positions the animal STILL LICKS, does lick activity still look like pre-stroke?

    Own-column value in the LICK window for `poststroke_lick`: 1.0 = that position's own pre-stroke
    lick signature, so this is "how much of the normal motor pattern is left", per position and per
    session. The impaired flag is behavioural (post-stroke response rate < 0.5), so the PRESERVED
    rows are the ones this question is about and the impaired ones are shown for contrast.
    """
    rates = response_rates()
    data = json.loads(SRC.read_text(encoding="utf-8"))
    res_all = data.get(window) or {}
    print(f"\n{'=' * 100}\n### {window} WINDOW — poststroke_lick, own-column value (1.0 = pre-stroke "
          f"lick at that position)\n")
    for an in ("PS92", "PS93", "PS94", "PS95"):
        res = res_all.get(an)
        if not res:
            continue
        cm = res["methods"][METH]["cross_matrix"]
        cs = res["methods"][METH].get("cross_by_session", {}).get("poststroke_lick", {})
        rr = rates.get(an, {})
        print(f"{an}")
        for p in POS:
            r = row(cm, "poststroke_lick", p)
            tag = "IMPAIRED" if rr.get(p, 1) < 0.5 else "licking "
            if p not in r:
                print(f"   {p:<13} {tag}  resp={rr.get(p, float('nan')):.2f}  no cell")
                continue
            b = best(r)
            per = []
            for sess in sorted(cs):
                c = (cs[sess].get(p) or {}).get(p) or {}
                if c.get("mean") is not None and (c.get("n") or 0) >= MIN_N:
                    per.append(f"{sess.split('_')[-1]}:{c['mean']:+.2f}")
            print(f"   {p:<13} {tag}  resp={rr.get(p, float('nan')):.2f}  own={r[p]:+.2f}  "
                  f"best={b}({r[b]:+.2f}){'' if b == p else '  <-- NOT own'}")
            if per:
                print(f"      per session: {'  '.join(per)}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--preserved":
        preserved(args[1] if len(args) > 1 else "lick")
    elif args and args[0] == "--sessions":
        by_session(args[1:] or ["ENL", "cue"])
    else:
        main(args or ["ENL", "cue"])

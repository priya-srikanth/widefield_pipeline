"""Pilot for the many-draws MATCHED frozen arm (Priya 2026-11-25): how many draws per session, and what they cost.

READ-ONLY ON EVERY STORE. Nothing is frozen and nothing is cached: the script sets WIDEFIELD_EPOCH_FROZEN=0 for the
draws (in memory) and WIDEFIELD_FROZEN_LOAD_ONLY=1, and REFUSES (raises) on any session-cache write or frozen-store
write -- a cold cache is reported, never filled from here. The only writes are the tables under ``--out`` (required,
must be a new or empty folder of your own).

    PYTHONPATH=<worktree> python scripts/matched_draws_pilot.py draws --out <dir> --animals PS93 PS94 --draws 400
    PYTHONPATH=<worktree> python scripts/matched_draws_pilot.py legacy --out <dir> --sessions-csv <epoch_5rmgap_..._sessions.csv>
    PYTHONPATH=<worktree> python scripts/matched_draws_pilot.py dryrun --out <dir> --draws 50

draws   For the chosen epoch's sessions of each animal: M independent label-seeded draws through the REAL collector
        (`grant_confusion._collect_5c_cached`, hooks MATCHED_ONLY_LABELS / MATCHED_DRAW_CAPTURE), each draw's matched
        recall at the far-contralateral spout and its gap (refit - matched). The refit column does not depend on the
        draw, so SD(gap) = SD(matched recall) exactly. Per session: the SD of one draw, the SD of the mean of N draws
        (analytic SD / sqrt(N); the draws are independent seeds) and, where M allows >= 8 groups, the empirical SD
        across non-overlapping groups of N; the wall time per draw.
legacy  The legacy single draw (old seed rule) through the store (load-only): every per-session, per-position gap of
        the collector must EQUAL the stored `_sessions.csv` of the existing render (exact, no tolerance).
dryrun  The frozen-store spec ids a render with WIDEFIELD_MATCHED_DRAWS=<N> would create (4 animals x windows x
        classes), and whether each already exists. Fits nothing.

Apply the render profile to the environment BEFORE launching (the queue's `render_profile.apply`); this script only
adds the read-only switches.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

#: (window, class) pairs of the matched family in the epoch render (ARMS of `epoch_grant_figures`)
PAIRS = (("precue", "working"), ("precue", "lick"), ("cue", "working"), ("cue", "lick"), ("lick", "lick"))
N_GRID = (1, 5, 20, 50, 100, 200)


def _read_only_guards():
    """Refuse every store write: the pilot reads the session cache and the frozen store, it never adds to them."""
    os.environ["WIDEFIELD_FROZEN_LOAD_ONLY"] = "1"
    from wfield_local import frozen_models, session_cache

    def _no_cache(fp, lab, kind, res, verbose=True):
        raise RuntimeError(f"[pilot] session-cache MISS for {lab}/{kind}: the pilot is cache-read-only "
                           f"(would have written {fp}). Check the profile / env against the render's.")

    def _no_freeze(spec, payload, meta, log=print):
        raise RuntimeError(f"[pilot] refusing to freeze {spec.get('animal')} {spec.get('kind')}/{spec.get('align')}")

    session_cache._store = _no_cache
    frozen_models._write = _no_freeze


def _out_dir(p) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _new(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} exists; the pilot never overwrites (new file names only)")
    return path


def _far_contra():
    """(position name, numeric code) of the far-contralateral spout, from the cohort's lesion side."""
    from wfield_local import epoch_accuracy as ea
    from wfield_local import epoch_figures as ef
    name = "far_R" if ef.lesion_side() == "L" else "far_L"
    return name, ea._code_of(name)


def _epoch_labels(gc, an, align, epoch):
    from wfield_local import epoch_figures as ef
    bd = gc._pooled_bundle(an, align)
    out = []
    for i, lab in enumerate(bd["kept"]):
        if i in bd["pre_i"]:
            if epoch == "pre":
                out.append(lab)
            continue
        if ef.epoch_of_day(an, int(gc._day(an, lab.split("_")[-1]))) == epoch:
            out.append(lab)
    return out


class _Timed(list):
    """The capture list, stamping each session's arrival time."""

    def __init__(self):
        super().__init__()
        self.t = []

    def append(self, x):
        super().append(x)
        self.t.append(time.perf_counter())


def cmd_draws(a):
    out = _out_dir(a.out)
    os.environ["WIDEFIELD_EPOCH_FROZEN"] = "0"                 # the draws are fitted in memory, nothing stored
    os.environ["WIDEFIELD_MATCHED_DRAWS"] = str(a.draws)
    _read_only_guards()
    from wfield_local import grant_confusion as gc
    import wfield_local
    print(f"[pilot] wfield_local = {wfield_local.__file__}", flush=True)
    pos, code = _far_contra()
    refit_t = []
    _rp = gc._refit_pred

    def _timed_refit(*args, **kw):
        t0 = time.perf_counter()
        r = _rp(*args, **kw)
        refit_t.append(time.perf_counter() - t0)
        return r

    gc._refit_pred = _timed_refit
    draw_rows, sess_rows = [], []
    for an in a.animals:
        t0 = time.perf_counter()
        labels = _epoch_labels(gc, an, a.align, a.epoch)
        t_bundle = time.perf_counter() - t0
        print(f"[pilot] {an}: {len(labels)} {a.epoch} session(s) {labels}; bundle {t_bundle:.0f} s", flush=True)
        cap = _Timed()
        gc.ANIMALS = (an,)
        gc.MATCHED_ONLY_LABELS = set(labels)
        gc.MATCHED_DRAW_CAPTURE = cap
        refit_t.clear()
        t0 = time.perf_counter()
        gc._collect_5c_cached.__wrapped__(a.align, a.variant, "paired_matched")
        stamps = [t0] + cap.t
        # the first session's interval also carries the frame fit (full + leave-one-out pre models): report apart
        for k, c in enumerate(cap):
            y, rf, hits = np.asarray(c["y"]), np.asarray(c["refit"]), np.asarray(c["hits"])
            m = (y == code) & (rf != gc.REFIT_UNAVAILABLE)
            wall = stamps[k + 1] - stamps[k]
            if not m.any():
                print(f"[pilot] {c['label']}: no scorable {pos} trial; skipped", flush=True)
                continue
            rec_d = hits[:, m].mean(1)                        # matched recall at far contra, per draw
            refit = float((rf[m] == y[m]).mean())
            all_d = hits[:, rf != gc.REFIT_UNAVAILABLE].mean(1)
            for d, (v, w) in enumerate(zip(rec_d, all_d)):
                draw_rows.append(dict(animal=an, label=c["label"], draw=d, seed=c["seeds"][d],
                                      matched_recall_farcontra=float(v), gap_farcontra=refit - float(v),
                                      matched_acc_all=float(w)))
            sd1 = float(rec_d.std(ddof=1))
            row = dict(animal=an, label=c["label"], epoch=a.epoch, align=a.align, variant=a.variant, position=pos,
                       n_trials=int(len(y)), n_farcontra=int(m.sum()), n_target=int(c["n_target"]),
                       n_draws=int(len(rec_d)), refit_recall=refit, matched_recall_mean=float(rec_d.mean()),
                       gap_mean=refit - float(rec_d.mean()), sd_one_draw=sd1,
                       min_draw=float(rec_d.min()), max_draw=float(rec_d.max()),
                       sd_all_positions_one_draw=float(all_d.std(ddof=1)),
                       wall_s=wall, first_of_animal=int(k == 0),
                       refit_s=float(refit_t[k]) if k < len(refit_t) else float("nan"))
            row["s_per_draw"] = (wall - row["refit_s"]) / len(rec_d) if k else float("nan")
            for n in N_GRID:
                row[f"sd_mean_N{n}"] = sd1 / np.sqrt(n)
                g = len(rec_d) // n
                row[f"sd_emp_N{n}"] = (float(rec_d[:g * n].reshape(g, n).mean(1).std(ddof=1)) if g >= 8
                                       else float("nan"))
            sess_rows.append(row)
            print(f"[pilot] {c['label']}: n {len(y)} ({int(m.sum())} {pos}), refit {refit:.3f}, matched "
                  f"{rec_d.mean():.3f} (one-draw SD {sd1:.3f}, range {rec_d.min():.2f}..{rec_d.max():.2f}), "
                  f"wall {wall:.0f} s", flush=True)
        gc.MATCHED_ONLY_LABELS = None
        gc.MATCHED_DRAW_CAPTURE = None
    tag = f"{a.align}_{a.variant}_{a.epoch}_M{a.draws}_{'-'.join(a.animals)}"
    _write(_new(out / f"pilot_draws_{tag}.csv"), draw_rows)
    _write(_new(out / f"pilot_sessions_{tag}.csv"), sess_rows)
    print(f"[pilot] wrote {out}", flush=True)
    return 0


def _write(path, rows):
    if not rows:
        print(f"[pilot] no rows for {path.name}", flush=True)
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_legacy(a):
    out = _out_dir(a.out)
    os.environ.pop("WIDEFIELD_MATCHED_DRAWS", None)
    if a.in_memory:
        os.environ["WIDEFIELD_EPOCH_FROZEN"] = "0"
    _read_only_guards()
    from wfield_local import epoch_accuracy as ea
    from wfield_local import epoch_figures as ef
    from wfield_local import grant_confusion as gc
    from wfield_local.grant_figures import CONF_LABELS
    import wfield_local
    print(f"[pilot] wfield_local = {wfield_local.__file__}", flush=True)
    if a.animals:
        gc.ANIMALS = tuple(a.animals)
    t0 = time.perf_counter()
    per_animal, _days = gc._collect_5c_cached.__wrapped__(a.align, a.variant, "paired_matched")
    print(f"[pilot] legacy collection {time.perf_counter() - t0:.0f} s, animals {sorted(per_animal)}", flush=True)
    short = dict(zip(CONF_LABELS, ea._short_labels()))
    got = {}
    for e in ef.PANELS:
        for q in CONF_LABELS:
            for an, v, s in ef.per_session_values(per_animal, e, lambda _an, _d, r, q=q: ea._gap_of(r, q),
                                                  with_session=True):
                if v is not None:
                    got[(e, short[q], an, s)] = v
    import hashlib
    dump = []
    for an, (pre, by_day) in sorted(per_animal.items()):
        items = ([(f"pre:{lab}", r) for lab, r in zip(getattr(pre, "labels", []), pre)] if pre is not None else [])
        items += [(f"day{d}", by_day[d]) for d in sorted(by_day)]
        for key, r in items:
            h = hashlib.sha256()
            for x in r:
                h.update(np.ascontiguousarray(np.asarray(x, np.float64)).tobytes())
            dump.append(dict(animal=an, record=key, n=len(r[0]), sha=h.hexdigest()[:20]))
    tag = f"{a.align}_{a.variant}{'_inmemory' if a.in_memory else '_store'}{a.tag}"
    _write(_new(out / f"legacy_records_{tag}.csv"), dump)
    print(f"[pilot] {len(dump)} records dumped", flush=True)
    if not a.sessions_csv:
        return 0
    want = {}
    with open(a.sessions_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not a.animals or r["animal"] in a.animals:
                want[(r["epoch"], r["position"], r["animal"], r["session"])] = float(r["value"])
    rows, n_bad = [], 0
    for k in sorted(set(got) | set(want)):
        g, w = got.get(k), want.get(k)
        same = g is not None and w is not None and float(repr(g)) == w
        n_bad += not same
        rows.append(dict(epoch=k[0], position=k[1], animal=k[2], session=k[3], stored=w, recomputed=g, equal=int(same)))
    _write(_new(out / f"legacy_check_{tag}.csv"), rows)
    print(f"[pilot] legacy check {a.align}/{a.variant}: {len(rows)} session x position values, "
          f"{len(rows) - n_bad} EQUAL, {n_bad} differ / missing (vs {a.sessions_csv})", flush=True)
    return 0


def cmd_compare(a):
    """Two record dumps (`legacy` of two code copies under the same env): every record's sha must be EQUAL."""
    def rd(f):
        with open(f, newline="", encoding="utf-8") as fh:
            return {(r["animal"], r["record"]): (r["n"], r["sha"]) for r in csv.DictReader(fh)}
    x, y = rd(a.a), rd(a.b)
    keys = sorted(set(x) | set(y))
    bad = [k for k in keys if x.get(k) != y.get(k)]
    print(f"[pilot] compare: {len(keys)} records, {len(keys) - len(bad)} EQUAL, {len(bad)} differ / missing")
    for k in bad[:20]:
        print("   ", k, x.get(k), y.get(k))
    return 1 if bad else 0


def cmd_dryrun(a):
    """The spec ids of a WIDEFIELD_MATCHED_DRAWS=<N> render. Builds each spec exactly as the collector does (same
    bundle, frame and selection), so the ids are the ones the render would ask for; fits nothing."""
    out = _out_dir(a.out)
    os.environ["WIDEFIELD_MATCHED_DRAWS"] = str(a.draws)
    _read_only_guards()
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    import wfield_local
    print(f"[pilot] wfield_local = {wfield_local.__file__}; store {frozen_models.local_dir()}", flush=True)
    seen = []

    class _Stop(Exception):
        pass

    def _find(spec):
        if spec.get("kind") == "matched_frozen":
            seen.append(spec)
            raise _Stop()
        return _real_find(spec)

    _real_find = frozen_models.find
    rows = []
    for align, variant in PAIRS:
        for an in (a.animals or gc.ANIMALS):
            seen.clear()
            gc_animals, gc.ANIMALS = gc.ANIMALS, (an,)
            frozen_models.find = _find
            # the collector's outer `except` reports and moves on: the spec is captured before the stop
            try:
                gc._collect_5c_cached.__wrapped__(align, variant, "paired_matched")
            except Exception:                                            # noqa: BLE001
                if not seen:             # a real failure before the spec was built (the camera gate re-raises)
                    raise
            finally:
                frozen_models.find = _real_find
                gc.ANIMALS = gc_animals
            if not seen:
                rows.append(dict(animal=an, align=align, variant=variant, slug="(no spec built)", spec_id="",
                                 exists="", n_pre_sessions="", folder=""))
                continue
            spec = seen[0]
            d, _o = _real_find(spec)
            rows.append(dict(animal=an, align=align, variant=variant, slug=frozen_models._slug(spec),
                             spec_id=frozen_models.spec_id(spec), exists=int(d is not None),
                             n_pre_sessions=len(spec["train_labels"]),
                             folder=str(frozen_models.local_dir() / an / frozen_models._slug(spec))))
            print(f"[pilot] {an} {align}/{variant}: {rows[-1]['slug']} exists={rows[-1]['exists']}", flush=True)
    _write(_new(out / f"dryrun_new_frozen_ids_draws{a.draws}{a.suffix}.csv"), rows)
    (_new(out / f"dryrun_env_draws{a.draws}{a.suffix}.json")).write_text(json.dumps(
        {k: v for k, v in sorted(os.environ.items()) if k.startswith(("WIDEFIELD_", "WFIELD_", "FROZEN_"))}, indent=1),
        encoding="utf-8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    for name in ("draws", "legacy", "dryrun"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="your own NEW output folder (required; no default)")
        p.add_argument("--animals", nargs="+", default=None)
        p.add_argument("--align", default="cue", choices=("precue", "cue", "lick"))
        p.add_argument("--variant", default="working", choices=("working", "lick"))
        if name == "draws":
            p.add_argument("--draws", type=int, default=400, help="M independent draws per session")
            p.add_argument("--epoch", default="acute")
        if name == "legacy":
            p.add_argument("--sessions-csv", default=None)
            p.add_argument("--tag", default="")
            p.add_argument("--in-memory", action="store_true", help="WIDEFIELD_EPOCH_FROZEN=0 instead of the store")
        if name == "dryrun":
            p.add_argument("--draws", type=int, required=True)
            p.add_argument("--suffix", default="")
    a = ap.parse_args(argv)
    if a.cmd == "draws" and not a.animals:
        ap.error("draws needs --animals")
    return {"draws": cmd_draws, "legacy": cmd_legacy, "dryrun": cmd_dryrun, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

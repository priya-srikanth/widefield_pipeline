"""O2 stages that test the set: `splithalf`, `loso` (leave one pre-stroke day out: the noise floor), `fitsession` (each session's own fit vs the frozen set) and `fitepoch` (pooled own fit per animal x epoch, + the two pre-day halves)."""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np

from .config import EPOCH_GROUPS
from .cp import match, ncp_hals, pooled_evr, project_rows, similarity
from .files import _atomic_savez, load_master, load_plan, npz_small, out_dir
from .project import task_basis
from .sample import gather_rows, session_group


def cmd_splithalf(sd: Path, rank: int, inits: int = 2) -> None:
    """IS THE SAMPLE ENOUGH? (Priya 10-23: "20 trials per position per animal is enough?") Split the fit rows in two
    disjoint halves -- the fit TRIALS by animal x spout position, alternately after a seeded shuffle (10 per position
    per animal each), every other row (pre-cue, between) at random -- fit rank R on each half (best of ``inits``
    starts) and compare: half A vs half B (matched spatial x spectral similarity, per factor and mean) and each half vs
    the full fit (``rank<R>.npz``). Halves that agree (~0.9+) = the full sample (twice each half) is comfortably enough.
    -> ``splithalf/rank<R>.json``."""
    pl = load_plan(sd)
    dst = sd / "splithalf" / f"rank{rank:02d}.json"
    X = np.load(sd / "fit_tensor.npy", mmap_mode="r")
    ix = np.load(sd / "fit_index.npz")
    rng = np.random.default_rng([int(pl["params"].get("seed", 0)), 7919, rank])
    half = rng.integers(0, 2, len(ix["key"])).astype(np.int8)
    tr = ix["kind"] == "trial"
    pos_of = {}
    for k in np.unique(ix["key"][tr]):
        pos_of[k] = npz_small(out_dir(sd) / f"{k}_trials.npz", ("position",))["position"]
    groups = {}
    for a, k, t in {(a, k, int(t)) for a, k, t in zip(ix["animal"][tr], ix["key"][tr], ix["trial"][tr])}:
        groups.setdefault((a, str(pos_of[k][t])), []).append((k, t))
    side = {}
    for g in sorted(groups):
        items = sorted(groups[g])
        for j, it in enumerate(rng.permutation(len(items))):
            side[items[it]] = j % 2
    half[tr] = [side[(k, int(t))] for k, t in zip(ix["key"][tr], ix["trial"][tr])]
    t0 = time.time()
    fits = []
    for h in (0, 1):
        Xh = np.ascontiguousarray(X[half == h])
        fs = [ncp_hals(Xh, rank, seed=500 + 10 * h + i, n_iter=pl["params"]["n_iter"]) for i in range(inits)]
        fits.append(min(fs, key=lambda f_: f_["rel_err"]))
        del Xh
        print(f"  half {'AB'[h]}: {int((half == h).sum())} rows, rel err {fits[-1]['rel_err']:.4f}", flush=True)
    full = np.load(sd / f"rank{rank:02d}.npz")
    _, cos_ab = match([fits[0]["A"], fits[0]["B"]], [fits[1]["A"], fits[1]["B"]])
    res = {"rank": rank, "inits": inits, "n_rows": [int((half == h).sum()) for h in (0, 1)],
           "n_trials": [int(sum(1 for v in side.values() if v == h)) for h in (0, 1)],
           "similarity_halves": float(np.prod(cos_ab, axis=0).mean()),
           "similarity_per_factor": np.prod(cos_ab, axis=0).round(4).tolist(),
           "similarity_half_to_full": [similarity([full["A"], full["B"]], [f_["A"], f_["B"]]) for f_ in fits],
           "rel_err_halves": [float(f_["rel_err"]) for f_ in fits], "minutes": (time.time() - t0) / 60}
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(res, indent=1))
    print(f"rank {rank} split half: halves similarity {res['similarity_halves']:.3f} (per factor min "
          f"{min(res['similarity_per_factor']):.3f}), half vs full {res['similarity_half_to_full'][0]:.3f} / "
          f"{res['similarity_half_to_full'][1]:.3f}, {res['minutes']:.0f} min -> {dst}")


def cmd_loso(sd: Path, job: int) -> None:
    """LEAVE ONE PRE-STROKE SESSION OUT (job = 1-based row of plan ``loso_jobs`` = (rank, day)): refit rank R without the
    day's fit rows (one random start, the fit's n_iter), then project the day's SESSION rows (the same rows `fitsession`
    scores every session on) with that fit (HELD OUT) and with the full fit (``rank<R>.npz``, IN FIT). The held-out pre
    EVR is the NOISE FLOOR the post-stroke frozen EVRs (and EVR gaps) are read against: how well a basis that never saw
    a pre-stroke day explains it. -> ``loso/rank<R>_<day>.json``."""
    pl = load_plan(sd)
    rank, day = pl["loso_jobs"][job - 1]
    dst = sd / "loso" / f"rank{rank:02d}_{day}.json"
    if dst.exists():
        print("done already", dst)
        return
    t0 = time.time()
    X = np.load(sd / "fit_tensor.npy", mmap_mode="r")
    keys = np.load(sd / "fit_index.npz")["key"]
    m = np.array([session_group(k) == day for k in keys])
    f = ncp_hals(np.ascontiguousarray(X[~m]), rank, seed=31 + rank, n_iter=pl["params"]["n_iter"])
    full = np.load(sd / f"rank{rank:02d}.npz")
    scales = json.loads((sd / "scales.json").read_text())
    tasks = [t for t in pl["tasks"] if session_group(t["key"]) == day]
    animal = tasks[0]["animal"]
    Xs = np.concatenate([gather_rows(sd, pl["session_rows"][t["key"]])[0] for t in tasks]) / scales[animal]
    _, e_out, n2 = project_rows(Xs, f["A"], f["B"])
    _, e_in, _ = project_rows(Xs, full["A"], full["B"])
    res = {"rank": rank, "day": day, "animal": animal, "epoch": "pre", "n_fit_rows_left_out": int(m.sum()),
           "n_session_rows": int(len(Xs)), "evr_held_out": pooled_evr(e_out, n2), "evr_in_fit": pooled_evr(e_in, n2),
           "similarity_to_full": similarity([full["A"], full["B"]], [f["A"], f["B"]]),
           "minutes": (time.time() - t0) / 60}
    order, cos = match([full["A"], full["B"]], [f["A"], f["B"]])
    res["cos_spatial"], res["cos_spectral"] = cos[0].tolist(), cos[1].tolist()
    dst.parent.mkdir(exist_ok=True)
    _atomic_savez(dst.with_suffix(".npz"), A=f["A"][:, order], B=f["B"][:, order], cos=cos, rank=rank, day=day)
    dst.write_text(json.dumps(res, indent=1))
    print(f"rank {rank} leave out {day}: EVR held out {res['evr_held_out']:.4f} vs in fit {res['evr_in_fit']:.4f} "
          f"({len(Xs)} session rows), similarity {res['similarity_to_full']:.3f}, {res['minutes']:.0f} min")


def cmd_fitsession(sd: Path, task: int, inits: int = 3, own_dir: str = "own") -> None:
    pl = load_plan(sd)
    t = pl["tasks"][task - 1]
    key, animal = t["key"], t["animal"]
    dst = sd / own_dir / f"{key}.npz"          # own_dir: e.g. "own8" for an 8-start rerun beside the 3-start own/
    if dst.exists():
        print("done already", dst)
        return
    A, B, scales, M = load_master(sd)
    A, B, basis, _ = task_basis(sd, t, A, B)
    R = A.shape[1]
    t0 = time.time()
    X, _ = gather_rows(sd, pl["session_rows"][key])
    X /= float(scales[animal])
    _, Ef, N2 = project_rows(X, A, B)
    evr_frozen = pooled_evr(Ef, N2)
    fits = [ncp_hals(X, R, seed=zlib.crc32(key.encode()) + k, n_iter=pl["params"]["n_iter"]) for k in range(inits)]
    f = fits[int(np.argmin([x["rel_err"] for x in fits]))]
    order, cos = match([A, B], [f["A"], f["B"]])
    _atomic_savez(dst, A_own=f["A"][:, order].astype(np.float32), B_own=f["B"][:, order].astype(np.float32),
                  evr_own=1 - f["rel_err"] ** 2, evr_frozen=evr_frozen, cos_spatial=cos[0], cos_spectral=cos[1],
                  rel_errs_own=np.array([x["rel_err"] for x in fits]), n_rows=len(X), spec_hash=str(M["spec_hash"]),
                  animal=animal, epoch=str(t["epoch"]), date=t["date"], basis=basis)
    print(f"{key} [{t['epoch']}, basis {basis}]: EVR own {1 - f['rel_err'] ** 2:.3f} vs frozen {evr_frozen:.3f}; matched cosine "
          f"spatial {cos[0].mean():.3f} spectral {cos[1].mean():.3f}; {len(X)} rows, {(time.time() - t0) / 60:.1f} min")


def epoch_fit_tasks(pl: dict) -> list[tuple[str, str, list[str]]]:
    """[(animal, group, task keys)] in a fixed order (animals sorted, then EPOCH_GROUPS); array task n = row n - 1."""
    out = []
    for an in sorted({t["animal"] for t in pl["tasks"]}):
        mine = sorted((t for t in pl["tasks"] if t["animal"] == an), key=lambda t: t["key"])
        pre_days = sorted({t["date"] for t in mine if t["epoch"] == "pre"})
        for g in EPOCH_GROUPS:
            if g in ("preA", "preB"):
                days = set(pre_days[(0 if g == "preA" else 1)::2])
                keys = [t["key"] for t in mine if t["epoch"] == "pre" and t["date"] in days]
            else:
                keys = [t["key"] for t in mine if t["epoch"] == g]
            out.append((an, g, keys))
    return out


# Priya 10-25: a DRIFT reference for the cohort shape readouts -- preA / preB interleave the pre days, so their floor
# leaves out slow change across days; early vs late pre days (same procedure) shows how much shape moves with time alone.
# Appended AFTER the 30 epoch tasks (31-40), so the existing task numbers are unchanged.
DRIFT_GROUPS = ("preEarly", "preLate")


def cohort_fit_tasks(pl: dict) -> list[tuple[str, str, dict[str, list[str]]]]:
    """[(tag, group, {animal: task keys})]: the COHORT (all animals) and each leave-one-animal-out set (``LOAO-<animal>``)
    x EPOCH_GROUPS, in a fixed order; array task n = row n - 1 (5 sets x 6 groups = 30 for four animals), then the same
    sets x DRIFT_GROUPS (tasks 31-40). The groups
    per animal are exactly `epoch_fit_tasks`' (preA / preB = that animal's alternating pre days)."""
    per: dict[str, dict[str, list[str]]] = {}
    for an, g, keys in epoch_fit_tasks(pl):
        per.setdefault(an, {})[g] = keys
    animals = sorted(per)
    for an in animals:                        # DRIFT halves: the pre DAYS in time order, first half vs second half
        days = sorted({k.split("_")[1] for k in per[an]["pre"]})
        early = set(days[:len(days) // 2])
        per[an]["preEarly"] = [k for k in per[an]["pre"] if k.split("_")[1] in early]
        per[an]["preLate"] = [k for k in per[an]["pre"] if k.split("_")[1] not in early]
    sets = [("COHORT", animals)] + [(f"LOAO-{an}", [a for a in animals if a != an]) for an in animals]
    return ([(tag, g, {a: per[a][g] for a in use}) for tag, use in sets for g in EPOCH_GROUPS]
            + [(tag, g, {a: per[a][g] for a in use}) for tag, use in sets for g in DRIFT_GROUPS])


def cmd_fitcohort(sd: Path, task: int, inits: int = 8, rows_per_animal: int = 5500,
                  own_dir: str = "own_epoch_cohort") -> None:
    """ONE own fit per EPOCH on all animals pooled (Priya 10-25: "shouldn't we just do a pooled animal per epoch fit?"),
    plus the leave-one-animal-out fits that show a result is not one animal's. Why: the per-animal x epoch fits split /
    merge the lick factors (PS95 subacute: lick_left spread over three own factors, best single column .74), and 16
    Hungarian matches per readout are 16 chances to mis-assign; the master itself is a cohort fit. Rows: the same
    ``rows_per_animal`` from every animal (drawn evenly over its sessions in the group, as `cmd_fitepoch`), each animal
    divided by ITS OWN scale (the master's normalisation), so no animal dominates; PS95 acute (1 session) gives what it
    has. preA / preB = every animal's alternating pre-day halves pooled -> their angle is the cohort noise floor."""
    pl = load_plan(sd)
    tag, g, by_animal = cohort_fit_tasks(pl)[task - 1]
    dst = sd / own_dir / f"{tag}_{g}.npz"
    if dst.exists():
        print("done already", dst)
        return
    _cohort_fit(sd, pl, by_animal, f"{tag}_{g}", dst, inits, rows_per_animal, animal=tag, group=g)


def _cohort_fit(sd: Path, pl: dict, by_animal: dict[str, list[str]], seed: str, dst: Path, inits: int,
                rows_per_animal: int, **meta) -> None:
    """The cohort fit itself (shared by `cmd_fitcohort` and the permutation null `cmd_fitcohortnull`, so the null is
    the SAME procedure): ``rows_per_animal`` rows per animal, even over its sessions, each animal / its own scale;
    ``inits`` HALS starts, best kept, columns aligned to the master; ``seed`` seeds the row draw and the starts."""
    A, B, scales, M = load_master(sd)
    R = A.shape[1]
    t0 = time.time()
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    parts, rows_of = [], {}
    for an in sorted(by_animal):
        keys = by_animal[an]
        per = rows_per_animal // max(len(keys), 1)
        n0 = sum(len(p) for p in parts)
        for k in keys:
            x = gather_rows(sd, pl["session_rows"][k])[0]
            parts.append(x[np.sort(rng.choice(len(x), min(per, len(x)), replace=False))] / float(scales[an]))
            del x
        rows_of[an] = sum(len(p) for p in parts) - n0
    X = np.concatenate(parts)
    del parts
    _, Ef, N2 = project_rows(X, A, B)
    fits = [ncp_hals(X, R, seed=zlib.crc32(seed.encode()) + k, n_iter=pl["params"]["n_iter"])
            for k in range(inits)]
    f = fits[int(np.argmin([x["rel_err"] for x in fits]))]
    order, cos = match([A, B], [f["A"], f["B"]])
    keys_all = [k for an in sorted(by_animal) for k in by_animal[an]]
    _atomic_savez(dst, A_own=f["A"][:, order].astype(np.float32), B_own=f["B"][:, order].astype(np.float32),
                  evr_own=1 - f["rel_err"] ** 2, evr_frozen=pooled_evr(Ef, N2), cos_spatial=cos[0],
                  cos_spectral=cos[1], rel_errs_own=np.array([x["rel_err"] for x in fits]), n_rows=len(X),
                  rows_per_animal=np.array([rows_of[a] for a in sorted(rows_of)]), animals=np.array(sorted(rows_of)),
                  keys=np.array(keys_all), spec_hash=str(M["spec_hash"]), **meta)
    print(f"{seed} ({', '.join(f'{a} {n}' for a, n in sorted(rows_of.items()))} rows): EVR own "
          f"{1 - f['rel_err'] ** 2:.3f} vs frozen {pooled_evr(Ef, N2):.3f}; cosine to master spatial {cos[0].mean():.3f} "
          f"spectral {cos[1].mean():.3f}; {(time.time() - t0) / 60:.1f} min")


NULL_EPOCHS = ("acute", "subacute", "chronic")


def cohort_null_tasks(pl: dict, n_perm: int) -> list[tuple[str, int, str]]:
    """[(epoch, permutation, side)] for `cmd_fitcohortnull`: epoch-major, then permutation, then side ("pre", "post");
    array task n = row n - 1 (3 x n_perm x 2)."""
    return [(ep, k, side) for ep in NULL_EPOCHS for k in range(n_perm) for side in ("pre", "post")]


def null_split(pl: dict, epoch: str, perm: int) -> dict[str, dict[str, list[str]]]:
    """One permutation's PSEUDO groups: per animal, its pre and ``epoch`` sessions pooled and their labels shuffled
    (seeded by epoch + permutation, so the "pre" and "post" tasks of one permutation draw the SAME split), keeping each
    animal's counts -> {"pre": {animal: keys}, "post": {animal: keys}}."""
    per: dict[str, dict[str, list[str]]] = {}
    for an, g, keys in epoch_fit_tasks(pl):
        per.setdefault(an, {})[g] = keys
    out: dict[str, dict[str, list[str]]] = {"pre": {}, "post": {}}
    for an in sorted(per):
        pre, post = list(per[an].get("pre", [])), list(per[an].get(epoch, []))
        if not pre or not post:
            continue
        pool = np.array(pre + post)
        rng = np.random.default_rng(zlib.crc32(f"null_{epoch}_{perm}_{an}".encode()))
        pool = pool[rng.permutation(len(pool))]
        out["pre"][an], out["post"][an] = sorted(pool[:len(pre)].tolist()), sorted(pool[len(pre):].tolist())
    return out


def cmd_fitcohortnull(sd: Path, task: int, n_perm: int = 100, inits: int = 8, rows_per_animal: int = 5500,
                      own_dir: str = "own_epoch_cohort_null") -> None:
    """PERMUTATION NULL for the cohort shape readouts (Priya 10-25: "yes set up the null distribution" -- the jackknife
    over 4 animals cannot test 7 / 7b). For each post epoch and permutation: within every animal, pre and that epoch's
    sessions are pooled and their labels shuffled (counts kept), and the cohort fit is redone on the pseudo-pre and the
    pseudo-post groups by the identical procedure (`_cohort_fit`). Any readout's change (post fit - pre fit) over the
    permutations is its null; the observed change is the same readout on the real COHORT_<epoch> vs COHORT_pre fits.
    The null includes drift between pre and post days (shuffling breaks time order), so "beyond the null" = the
    sessions differ, by stroke or by time."""
    pl = load_plan(sd)
    ep, k, side = cohort_null_tasks(pl, n_perm)[task - 1]
    dst = sd / own_dir / f"{ep}_p{k:03d}_{side}.npz"
    if dst.exists():
        print("done already", dst)
        return
    split = null_split(pl, ep, k)[side]
    _cohort_fit(sd, pl, split, f"null_{ep}_{k}_{side}", dst, inits, rows_per_animal, animal="COHORT", group=side,
                null_epoch=ep, perm=k)


def cmd_fitepoch(sd: Path, task: int, inits: int = 8, n_rows: int = 11000, own_dir: str = "own_epoch") -> None:
    """ONE own fit per animal x epoch on that epoch's sessions pooled (Priya 10-24: per-session own fits are too noisy to
    read factor shape -- pre own-vs-master angles 10-50 deg). Every fit gets the SAME number of rows (``n_rows``, ~2
    sessions; drawn evenly over the group's sessions from their `session_rows`), so a fit's noise does not depend on
    how many sessions the epoch has; a group with fewer rows uses all of them (PS95 acute = 1 session) and says so.
    preA / preB = disjoint alternating halves of the pre days -> their angle is the noise floor for pre vs post."""
    pl = load_plan(sd)
    an, g, keys = epoch_fit_tasks(pl)[task - 1]
    dst = sd / own_dir / f"{an}_{g}.npz"
    if dst.exists():
        print("done already", dst)
        return
    A, B, scales, M = load_master(sd)
    R = A.shape[1]
    t0 = time.time()
    rng = np.random.default_rng(zlib.crc32(f"{an}_{g}".encode()))
    per = n_rows // len(keys)                        # even over sessions; a short session gives what it has
    parts = []
    for k in keys:                                   # subsample as each session is read (a row is ~0.4 MB: nine
        x = gather_rows(sd, pl["session_rows"][k])[0]    # whole pre sessions at once would be ~20 GB)
        parts.append(x[np.sort(rng.choice(len(x), min(per, len(x)), replace=False))])
        del x
    X = np.concatenate(parts) / float(scales[an])
    del parts
    _, Ef, N2 = project_rows(X, A, B)
    fits = [ncp_hals(X, R, seed=zlib.crc32(f"{an}_{g}".encode()) + k, n_iter=pl["params"]["n_iter"])
            for k in range(inits)]
    f = fits[int(np.argmin([x["rel_err"] for x in fits]))]
    order, cos = match([A, B], [f["A"], f["B"]])
    _atomic_savez(dst, A_own=f["A"][:, order].astype(np.float32), B_own=f["B"][:, order].astype(np.float32),
                  evr_own=1 - f["rel_err"] ** 2, evr_frozen=pooled_evr(Ef, N2), cos_spatial=cos[0],
                  cos_spectral=cos[1], rel_errs_own=np.array([x["rel_err"] for x in fits]), n_rows=len(X),
                  n_rows_target=n_rows, keys=np.array(keys), spec_hash=str(M["spec_hash"]), animal=an, group=g)
    print(f"{an} {g} ({len(keys)} sessions): EVR own {1 - f['rel_err'] ** 2:.3f} vs frozen {pooled_evr(Ef, N2):.3f}; "
          f"cosine to master spatial {cos[0].mean():.3f} spectral {cos[1].mean():.3f}; {len(X)} rows"
          f"{' (FEWER than target)' if len(X) < 0.95 * n_rows else ''}, {(time.time() - t0) / 60:.1f} min")

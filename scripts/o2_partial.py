"""The frozen-encoder PARTIAL REFITS (`scripts/partial_refit.py`) on HMS O2 (Priya 11-09 ~22:50: the three refits are the
~28 h critical path of the desktop glue re-run). One session = one SLURM array task; O2 never reads the share.

    python scripts/o2_partial.py bundle  --step b_partial_frA_g --name <name> --user ps150 [--jobs 2] [LABELS ...]
    python scripts/o2_partial.py check   --step b_partial_frA_g --name <name> --out <scratch dir> LABEL
    python scripts/o2_partial.py collect --step b_partial_frA_g --name <name>

STEPS (the glue re-run queue's three partial steps, env tokens copied from queue_daqlk20.sh `apply_env`, applied on top
of the render profile `cohort1_rebuild_20261103`): each step's frozen run must be DONE (its fit/pre_stats.pkl +
fit/templates.pkl + joint/sessions/) before its bundle can be built.

WHAT ONE TASK NEEDS (`partial_refit.job` for one label): the animal's frozen fit (fits.pkl, computed HERE from the run's
fit/pre_stats.pkl exactly as `partial_refit.frozen_fits`), the animal's templates (fit/templates.pkl), and everything
`frozen_encoder.session_design` reads for that session. The design is rebuilt 13 times per session (the fit + the
shuffled / permuted nulls, which patch `raw_blocks` / `spout_split` / `te_lick_parts`), but every share read sits in a
loader that the nulls do not patch (`partial_refit._reuse_inputs` relies on the same fact). So the bundle records, per
session, the RETURN VALUES of those loaders (`LOADERS`), keyed by their arguments, and the O2 runner replaces the loaders
by lookups (a missing key raises: never a silent share read). The FROZEN_DESIGN=te lick families also read a few small
tables outside the loaders (licks_all.csv, the landing calibration, the behavior-session folder listing): those are
recorded by `pandas.read_csv` / `Path.glob` recorders and replayed by path suffix.
  imaging/<label>.pkl            `combined_encoding.load_imaging(label)`, reduced to the STAT-mask pixels (the only
                                 pixels `session_design` reads for Allen-area targets: identical area means), shared by
                                 every step of the bundle (same masks + hemo variant, checked)
  runs/<step>/inputs/<label>.pkl  the other loaders' values + recorded tables, a digest of the design built here
  runs/<step>/fit/templates.pkl, runs/<step>/partial_refit/fits.pkl (+ .mask_sig), env.json, run.json, labels.txt
Guarantees as `scripts/o2_encoding.py`: env.json = every setting (profile env + COMBINED_ENCODING_* / OROFACIAL_* /
FROZEN_*; this machine's store / cache paths excluded); the runner clears and applies it BEFORE importing wfield_local and
REFUSES unless the bundled code computes the recorded identity (`o2_bundle.identity()` + the frozen design: design name,
running-model tag, TE / lick-spout / FR variables); the analysis-mask digest is recorded in bundle.json and in every
pickle (`o2_bundle.verify_masks` at bundle / collect time, `o2_bundle.check_inputs` on O2); code/ = `git archive` of HEAD
(wfield_local, configs, scripts), refused when dirty; each task re-checks its design digest against the bundle's.
check: the bundled task (a clean subprocess with ONLY env.json, the bundle's code, share reads audited) and the desktop
code path (`partial_refit.job` on the share, same fits) for one label; the session scores must agree (< 1e-9).
collect: runs/<step>/partial_refit/sessions/*.json -> <run>/partial_refit/sessions/ (refuses foreign files there),
fits.pkl + its mask sidecar, a provenance json, then `partial_refit.py --plot-only` (gain.csv, boot*.csv, figures) under
the step's env -- exactly the desktop step's outputs. It never writes the queue marker; it prints the instruction.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROFILE = "cohort1_rebuild_20261103"
#: queue_daqlk20.sh step -> (frozen run tag, env tokens, upstream frozen step, desktop --jobs). Tokens as `apply_env`.
STEPS = {
    "b_partial_frA_g": ("gate-attempted_v7g3_t3d_spont_frA", "frA", "f_frA_g", 4),
    "a_partial_ls_g": ("gate-attempted_v7g3_t3d_spont_frA_ls", "frozen", "f_frA_ls_g", 4),
    "t_partial_te_g": ("gate-attempted_v7g3_t3d_frA_te", "frA FROZEN_DESIGN=te FROZEN_LICKS_ALL=none", "t_frame_te_g", 3),
}
QUEUE_LOGS = ("C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/"
              "scratchpad/render_daqlk/logs")
INPUT_CACHE = "C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104"     # the queue's INCACHE
EXEC3D = "C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/pose3d_kin_licks_snapshot_20261023.csv.gz"
EXTRA_PREFIXES = ("FROZEN_",)
#: unlike the combined-encoding bundle no setting is "per submit" here: the queue's env (incl. COMBINED_ENCODING_ONLY /
#: SUFFIX) is recorded as is
LOCAL_ONLY_EXTRA = frozenset({"FROZEN_OUT_ROOT"})


# ------------------------------------------------------------------------------------------------------------ env
def env_noFR(env) -> None:
    env.update({"OROFACIAL_GATE": "attempted", "COMBINED_ENCODING_LICKS": "combined", "COMBINED_ENCODING_ONLY": "full",
                "COMBINED_ENCODING_SUFFIX": "_gate-attempted", "COMBINED_ENCODING_TONGUE": "3d",
                "COMBINED_ENCODING_EXEC3D_TABLE": EXEC3D, "FROZEN_LICKS_ALL": "split"})
    for k in ("COMBINED_ENCODING_LICKS_ALL", "COMBINED_ENCODING_EXEC", "COMBINED_ENCODING_TARGET3D_TABLE",
              "FROZEN_OUT_ROOT", "FROZEN_DESIGN", "FROZEN_RUNNING", "FROZEN_RUN_KNOT0", "FROZEN_PUSH", "FROZEN_FR",
              "FROZEN_FR_SPLIT", "FROZEN_LICKSPOUT", "FROZEN_LICKLAND", "FROZEN_FR_LAGS", "FROZEN_PIX_MUSALL",
              "COMBINED_ENCODING_FR", "COMBINED_ENCODING_FR_GROUPS"):
        env.pop(k, None)


def apply_tokens(tokens: str, env=None) -> None:
    """queue_daqlk20.sh `apply_env`: base | noFR | frA | frozen (= frA + FROZEN_LICKSPOUT=1) | VAR=value | unset:VAR."""
    env = os.environ if env is None else env
    for e in tokens.split():
        if e == "base":
            continue
        if e in ("noFR", "frA", "frozen"):
            env_noFR(env)
            if e in ("frA", "frozen"):
                env.update({"FROZEN_FR": "1", "COMBINED_ENCODING_FR": "master_pre_v1",
                            "COMBINED_ENCODING_FR_GROUPS": "master_v1:A"})
            if e == "frozen":
                env["FROZEN_LICKSPOUT"] = "1"
        elif e.startswith("unset:"):
            env.pop(e[6:], None)
        elif "=" in e:
            k, v = e.split("=", 1)
            env[k] = v
        else:
            raise SystemExit(f"bad env token {e}")


def apply_step_env(step: str, input_cache: str = INPUT_CACHE) -> None:
    """Desktop: the render profile, then the step's tokens, as the queue lane does (profile_env + apply_env)."""
    from wfield_local import render_profile
    for k in [k for k in os.environ if k.startswith(EXTRA_PREFIXES)]:
        del os.environ[k]                       # the profile clears WIDEFIELD_* / COMBINED_ENCODING_*, not FROZEN_*
    render_profile.apply(PROFILE)
    apply_tokens(STEPS[step][1])
    if input_cache:
        os.environ["WFIELD_INPUT_CACHE"] = input_cache


def settings() -> dict:
    from wfield_local import o2_bundle
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(o2_bundle.RECORD_PREFIXES + EXTRA_PREFIXES) and v
            and k not in o2_bundle.LOCAL_ONLY and k not in LOCAL_ONLY_EXTRA}


def identity() -> dict:
    """`o2_bundle.identity()` + what the frozen design computes to in this env (module constants read at import)."""
    from wfield_local import frozen_encoder as fe
    from wfield_local import o2_bundle
    return {**o2_bundle.identity(), "design": fe.design_name(), "frozen_run_tag": fe._run_tag(), "te": bool(fe.TE),
            "lickspout": bool(fe.LICKSPOUT), "fr_vars": list(fe.FR_VARS)}


def verify_identity(recorded: dict) -> None:
    from wfield_local import o2_bundle
    now = identity()
    base = {k: recorded[k] for k in ("run_tag", "selection", "area_basis")}
    o2_bundle.verify(base)
    bad = {k: (recorded.get(k), now[k]) for k in now if k not in base and recorded.get(k) != now[k]}
    if bad:
        raise o2_bundle.BundleMismatch("bundle settings do not reproduce the recorded frozen design: "
                                       + "; ".join(f"{k}: recorded {a!r} vs computed {b!r}" for k, (a, b) in bad.items()))


# ------------------------------------------------------------------------------------- loader record / replay
def _fe():
    from wfield_local import frozen_encoder as fe
    return fe


def _targets() -> list:
    """(owner object, attribute) of every loader: their return values are recorded / replayed."""
    from wfield_local import config, dlc_frames
    fe = _fe()
    ce = fe._ce()
    return [(ce, "load_imaging"), (fe, "load_behaviour"), (fe, "_fr_columns"), (config, "load_sessions"),
            (fe, "lick_landing_weights"), (fe, "_kin_all"), (dlc_frames, "lick_onsets")]


def _argkey(x):
    import numpy as np
    from wfield_local.paths import PathResolver
    if isinstance(x, PathResolver) or x is None:
        return "-"
    if isinstance(x, np.ndarray):
        return ("nd", x.shape, str(x.dtype), hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:16])
    if isinstance(x, (str, int, float, bool, tuple, list)):
        return repr(x)
    return type(x).__name__                      # e.g. the Behaviour of lick_landing_weights (fixed per session)


def call_key(name: str, args, kwargs) -> str:
    licks = os.environ.get("COMBINED_ENCODING_LICKS")   # load_behaviour's cam1 fallback pops it (session_design)
    return repr((name, tuple(_argkey(a) for a in args), tuple(sorted((k, _argkey(v)) for k, v in kwargs.items())), licks))


def _tail(p) -> list:
    return [x for x in str(p).replace("\\", "/").lower().split("/") if x not in ("", ".")]


SHARE_MARKS = ("m:", "n:", "/microscope/", "wfield_cache", "//")


def _is_ext(p) -> bool:
    s = str(p).replace("\\", "/").lower()
    return s.startswith(("m:", "n:", "//")) or any(m in s for m in SHARE_MARKS[2:4])


class Recorder:
    """Desktop: wrap the loaders (top-level calls only) and record {key: value}; also top-level `pandas.read_csv` /
    `Path.glob` on share / cache paths ({suffix path: bytes} / {dir suffix|pattern: names})."""

    def __init__(self, label: str):
        self.label, self.memo, self.files, self.globs, self.depth, self.orig = label, {}, {}, {}, 0, []

    def __enter__(self):
        import pathlib

        import pandas as pd
        for owner, name in _targets():
            f = getattr(owner, name)
            self.orig.append((owner, name, f))
            setattr(owner, name, self._wrap(name, f))
        rc, gl = pd.read_csv, pathlib.Path.glob
        self.orig += [(pd, "read_csv", rc), (pathlib.Path, "glob", gl)]
        me = self

        def read_csv(path, *a, **k):
            if me.depth == 0 and isinstance(path, (str, os.PathLike)) and _is_ext(path):
                me.files["/".join(_tail(path))] = Path(path).read_bytes()
            return rc(path, *a, **k)

        def glob(self_, pattern, *a, **k):
            out = list(gl(self_, pattern, *a, **k))
            if me.depth == 0 and _is_ext(self_):
                me.globs["/".join(_tail(self_)) + "|" + pattern] = [p.name for p in out]
            return iter(out)
        pd.read_csv, pathlib.Path.glob = read_csv, glob
        return self

    def _wrap(self, name, f):
        me = self

        def g(*a, **k):
            if me.depth:
                return f(*a, **k)
            me.depth += 1
            try:
                v = f(*a, **k)
            finally:
                me.depth -= 1
            me.memo[call_key(name, a, k)] = (name, _reduce(name, v, me.label))
            return v
        return g

    def __exit__(self, *exc):
        for owner, name, f in reversed(self.orig):
            setattr(owner, name, f)


def _reduce(name, v, label):
    """The recorded value: load_imaging keeps only the stat-mask pixels (all `session_design` reads for Allen-area
    targets: the same rows in the same order -> identical area means); _kin_all keeps this session's rows (its only
    reader filters by session first)."""
    import numpy as np
    if name == "load_imaging":
        V, ft, Ub, flat, parea, names, shape, pstat = v
        ps = np.asarray(pstat, bool)
        return ("imaging_reduced", (V, ft, np.ascontiguousarray(Ub[ps]), np.asarray(flat)[ps], np.asarray(parea)[ps],
                                    names, shape, np.ones(int(ps.sum()), bool)))
    if name == "_kin_all":
        from wfield_local import config
        an, date = config.animal_of(label), config.date8(label.split("_")[-1])
        return v[v.session == f"{an}_{date}"].copy()
    return v


def _expand(v):
    return v[1] if isinstance(v, tuple) and len(v) == 2 and v[0] == "imaging_reduced" else v


class Replay:
    """O2: the loaders return the recorded values (unknown key -> RuntimeError); recorded tables / listings are served
    by path suffix (>= 2 components, unique)."""

    def __init__(self, memo: dict, files: dict, globs: dict):
        self.memo, self.files, self.globs, self.orig = memo, files, globs, []

    @staticmethod
    def _match(keys, path_tail):
        for k in range(min(len(path_tail), 6), 1, -1):
            suf = path_tail[-k:]
            hit = [x for x in keys if x.split("/")[-k:] == suf]
            if len(hit) == 1:
                return hit[0]
        return None

    def __enter__(self):
        import copy
        import pathlib

        import pandas as pd
        me = self
        for owner, name in _targets():
            self.orig.append((owner, name, getattr(owner, name)))

            def g(*a, _n=name, **k):
                key = call_key(_n, a, k)
                if key not in me.memo:
                    raise RuntimeError(f"o2_partial: {_n} call not in the bundle (would read the share): {key[:300]}")
                v = _expand(me.memo[key][1])
                return copy.deepcopy(v) if _n == "load_sessions" else v
            setattr(owner, name, g)
        rc, gl = pd.read_csv, pathlib.Path.glob
        self.orig += [(pd, "read_csv", rc), (pathlib.Path, "glob", gl)]

        def read_csv(path, *a, **k):
            if isinstance(path, (str, os.PathLike)):
                hit = me._match(list(me.files), _tail(path))
                if hit is not None:                  # a buffer has no name: say the compression
                    return rc(io.BytesIO(me.files[hit]), *a, **{**k, **({"compression": "gzip"}
                                                                        if hit.endswith(".gz") else {})})
            return rc(path, *a, **k)

        def glob(self_, pattern, *a, **k):
            keys = [x.rsplit("|", 1)[0] for x in me.globs if x.rsplit("|", 1)[1] == pattern]
            hit = me._match(keys, _tail(self_))
            if hit is not None:
                return iter([self_ / n for n in me.globs[f"{hit}|{pattern}"]])
            return gl(self_, pattern, *a, **k)
        pd.read_csv, pathlib.Path.glob = read_csv, glob
        return self

    def __exit__(self, *exc):
        for owner, name, f in reversed(self.orig):
            setattr(owner, name, f)


def design_digest(d: dict) -> str:
    import numpy as np
    h = hashlib.sha1()
    for k in ("X", "Rd"):
        h.update(np.ascontiguousarray(d[k]).tobytes())
    h.update(repr(list(d["col_var"])).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------- bundle
def _git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _bundle_dir(name: str) -> Path:
    from wfield_local import rebuild_root, run_mode
    r = rebuild_root.root()
    if r is None:
        raise SystemExit("no rebuild root active (the profile sets it)")
    run_mode.register_data_store(Path(r) / "o2")
    return Path(r) / "o2" / name


def _run_dir(step: str) -> Path:
    from fr_musall import E
    return E / f"frozen_encoder_{STEPS[step][0]}"


def capture_worker(job):
    """One session: run `session_design` under the Recorder -> imaging pickle (if missing) + inputs pickle."""
    label, run_dir, rdir, imdir, mask_sig = job
    from wfield_local import config, mask_keys
    fe = _fe()
    if mask_keys.signature() != mask_sig:
        raise RuntimeError(f"{label}: worker masks differ from the bundle's {mask_sig}")
    an = config.animal_of(label)
    tpl = fe._templates_of(str(Path(run_dir) / "fit" / "templates.pkl"), an)
    with Recorder(label) as rec:
        d = fe.session_design(label, tpl)
    imaging = {k: v for k, v in rec.memo.items() if v[0] == "load_imaging"}
    other = {k: v for k, v in rec.memo.items() if v[0] != "load_imaging"}
    ip = Path(imdir) / f"{label}.pkl"
    from wfield_local import combined_encoding as ce
    hemo = ce.hemo_dir()
    if ip.exists():
        old = pickle.loads(ip.read_bytes())
        if old[mask_keys.FIELD] != mask_sig or old["hemo"] != hemo or set(old["memo"]) != set(imaging):
            raise RuntimeError(f"{ip}: built on other masks / hemo / keys; use a new bundle name")
    else:
        tmp = ip.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({mask_keys.FIELD: mask_sig, "hemo": hemo, "label": label, "memo": imaging},
                                     protocol=5))
        os.replace(tmp, ip)
    out = {mask_keys.FIELD: mask_sig, "label": label, "memo": other, "files": rec.files, "globs": rec.globs,
           "design": design_digest(d), "n_frames": int(len(d["X"])), "n_cols": int(d["X"].shape[1])}
    dst = Path(rdir) / "inputs" / f"{label}.pkl"
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out, protocol=5))
    os.replace(tmp, dst)
    return {"label": label, "MB": round((dst.stat().st_size + ip.stat().st_size) / 1e6),
            "loaders": sorted({v[0] for v in rec.memo.values()}), "files": sorted(rec.files), "globs": sorted(rec.globs)}


def _upstream_done(step: str) -> bool:
    return (Path(QUEUE_LOGS) / f"{STEPS[step][2]}.done").exists()


def cmd_bundle(a) -> Path:
    import datetime

    from wfield_local import analysis_kit as ak
    from wfield_local import mask_keys, o2_bundle, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    import partial_refit as pr
    tag, tokens, up, _jobs = STEPS[a.step]
    fe = _fe()
    fe.check_tag(tag)
    if not _upstream_done(a.step) and not a.ignore_upstream:
        raise SystemExit(f"{a.step}: upstream {up} has no .done marker in {QUEUE_LOGS}")
    run = _run_dir(a.step)
    for f in ("fit/pre_stats.pkl", "fit/templates.pkl"):
        if not (run / f).exists():
            raise SystemExit(f"{run / f} missing: {up} not finished")
    repo = Path(__file__).resolve().parents[1]
    dirty = _git(repo, "status", "--porcelain", "--", "wfield_local", "configs", "scripts")
    if dirty:
        raise SystemExit(f"uncommitted changes in {repo} (the bundle archives HEAD):\n{dirty}")
    commit = _git(repo, "rev-parse", "HEAD")
    d = _bundle_dir(a.name)
    writeguard.assert_writable(d)
    rdir = d / "runs" / a.step
    for sub in (d / "imaging", d / "logs", rdir / "inputs", rdir / "fit", rdir / "partial_refit", rdir / "tables"):
        sub.mkdir(parents=True, exist_ok=True)
    ident = identity()
    mask_sig = mask_keys.signature()
    meta_f = d / "bundle.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if meta:
        try:
            o2_bundle.verify_masks(meta, mask_sig)
        except o2_bundle.BundleMismatch as ex:
            raise SystemExit(f"refusing to extend {d}: {ex}") from None
    rmeta_f = rdir / "run.json"
    if rmeta_f.exists() and json.loads(rmeta_f.read_text())["identity"] != ident:
        raise SystemExit(f"{rdir} holds a bundle of another env: {json.loads(rmeta_f.read_text())['identity']} vs {ident}")
    # frozen fit + templates (desktop computation, exactly partial_refit.frozen_fits)
    fits = rdir / "partial_refit" / "fits.pkl"
    if fits.exists():
        mask_keys.check_sidecar(fits)
    else:
        print("frozen fits ...", flush=True)
        pr.frozen_fits(run, dest=fits)
    shutil.copy2(run / "fit" / "templates.pkl", rdir / "fit" / "templates.pkl")
    labels = a.labels or sorted(p.stem for p in (run / "joint" / "sessions").glob("*.json"))
    todo = [(lab, str(run), str(rdir), str(d / "imaging"), mask_sig) for lab in labels
            if not (rdir / "inputs" / f"{lab}.pkl").exists()]
    print(f"{len(todo)} of {len(labels)} sessions to capture", flush=True)
    got, fail = ak.fan_sessions(todo, capture_worker, jobs=a.jobs, key=lambda j: j[0], label="capture")
    for it, err in fail:
        print(f"  FAILED {it[0]}: {err}")
    rep = [r for _it, r in got]
    if rep:
        print(f"  loaders recorded: {sorted({x for r in rep for x in r['loaders']})}; tables: "
              f"{sorted({x.split('/')[-1] for r in rep for x in r['files']})}; listings: {sum(len(r['globs']) for r in rep)}")
    ok = sorted(p.stem for p in (rdir / "inputs").glob("*.pkl"))
    (rdir / "labels.txt").write_text("".join(f"{x}\n" for x in ok), encoding="utf-8", newline="\n")
    env = {}
    for k, v in settings().items():                       # a file setting travels inside the bundle ('@' = run dir)
        if Path(v).is_absolute() and Path(v).is_dir():
            raise SystemExit(f"{k}={v} names a directory: O2 cannot see it")
        if Path(v).is_absolute() and Path(v).is_file():
            shutil.copy2(v, rdir / "tables" / Path(v).name)
            v = f"@tables/{Path(v).name}"
        env[k] = v
    (rdir / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8", newline="\n")
    rmeta = {"step": a.step, "tag": tag, "tokens": tokens, "upstream": up, "identity": ident, "run_dir": str(run),
             "n_sessions": len(ok), "commit": commit,
             "pre_stats": {"size": (run / "fit" / "pre_stats.pkl").stat().st_size,
                           "mtime": (run / "fit" / "pre_stats.pkl").stat().st_mtime},
             "created": datetime.datetime.now().isoformat(timespec="seconds")}
    rmeta_f.write_text(json.dumps(rmeta, indent=1), encoding="utf-8", newline="\n")
    from wfield_local import rebuild_root
    meta.update({"name": a.name, "commit": commit, "code_root": str(repo), "profile": PROFILE,
                 "rebuild_root": str(rebuild_root.root() or ""), o2_bundle.MASK_FIELD: mask_sig,
                 "runs": sorted(set(meta.get("runs", [])) | {a.step})})
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    code = d / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir()
    import tempfile
    with tempfile.TemporaryDirectory() as td:      # archive locally (git archive -o onto the share failed 0xC0000142)
        tar = Path(td) / "code.tar"
        subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), commit, "wfield_local",
                        "configs", "scripts"], check=True)
        shutil.unpack_archive(str(tar), str(code))
    (d / "partial_runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    rv = PathResolver()
    b = f"{oi.scratch(a.user)}/o2/{a.name}"
    share_b = oi.o2_path(d, rv)
    sb = SBATCH.format(name=a.name, b=b, mem=a.mem, time=a.time)
    (d / "run_partial.sbatch").write_text(sb, encoding="utf-8", newline="\n")
    write_commands(d, meta, b, share_b, a.user)
    print(f"{len(ok)} sessions bundled for {a.step} -> {d}")
    return d


RUNNER = r'''"""Runs on O2: scripts/partial_refit.py for ONE session from the bundle (o2_partial.py). STEP + SLURM_ARRAY_TASK_ID
from the environment; or: python partial_runner.py STEP --label LABEL [--fits F] [--audit]."""
import json, os, sys
from pathlib import Path

b = Path(__file__).resolve().parent
args = sys.argv[1:]
step = args[0] if args and not args[0].startswith("--") else os.environ["STEP"]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i] in ("--label", "--fits")}
r = b / "runs" / step
env = json.loads((r / "env.json").read_text())
for k in [k for k in os.environ if k.startswith(("WIDEFIELD_", "WFIELD_", "LOCANMF_", "COMBINED_ENCODING_", "OROFACIAL_",
                                                 "FROZEN_"))]:
    del os.environ[k]                            # ONLY the bundle's settings apply
for k, v in env.items():
    os.environ[k] = str(r / v[1:]) if isinstance(v, str) and v.startswith("@") else v
sys.path[:0] = [str(b / "code"), str(b / "code" / "scripts")]
import o2_partial
if "--label" in opt:
    label = opt["--label"]
else:
    labels = [x.strip() for x in (r / "labels.txt").read_text().splitlines() if x.strip()]
    label = labels[int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1]
raise SystemExit(o2_partial.run_task(b, step, label, fits=opt.get("--fits"), audit="--audit" in args))
'''

SBATCH = """#!/bin/bash -l
#SBATCH -J part_{name}
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -o {b}/logs/%x_%A_%a.out
#SBATCH -e {b}/logs/%x_%A_%a.err
# STEP via sbatch --export=ALL,STEP=<queue step> (see COMMANDS.md). Login shell + conda activate (as run_enc.sbatch).
set -eo pipefail
module load conda/miniforge3/26.9.4-1
conda activate lp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "task $SLURM_ARRAY_TASK_ID STEP=$STEP on $(hostname)"
python {b}/partial_runner.py "$STEP"
"""


def write_commands(d: Path, meta: dict, b: str, share_b: str, user: str) -> None:
    runs = []
    for step in meta["runs"]:
        rm = json.loads((d / "runs" / step / "run.json").read_text())
        runs.append((step, rm))
    blocks = []
    for step, rm in runs:
        n = rm["n_sessions"]
        blocks.append(f"""### {step}  (frozen run {rm['tag']}; env tokens "{rm['tokens']}"; {n} sessions)
Login node, ONE task first:
```bash
sbatch --array=1 --export=ALL,STEP={step} {b}/run_partial.sbatch
tail -n 3 {b}/logs/part_{meta['name']}_*_1.out      # must end in "{step} <label>: ... done in ... min"
```
then all (task 1 is skipped as done):
```bash
sbatch --array=1-{n}%60 --export=ALL,STEP={step} {b}/run_partial.sbatch
```
Transfer node when done:
```bash
ls {b}/runs/{step}/partial_refit/sessions/*.json | wc -l      # expect {n}
rsync -ah {b}/runs/{step}/partial_refit/sessions/ {share_b}/runs/{step}/partial_refit/sessions/
```
Desktop (worktree at this bundle's code, PYTHONPATH=<worktree>;<worktree>/scripts):
```bash
python scripts/o2_partial.py collect --step {step} --name {meta['name']}
```
""")
    txt = f"""# O2 frozen-encoder partial refits: {meta['name']}

code {meta['commit']} ({Path(meta['code_root']).name}); profile {meta['profile']}; analysis masks {meta['mask_sig']}
Per step: runs/<step>/env.json (applied by the runner before wfield_local is imported; it REFUSES unless the bundled
code computes the recorded run tag / selection / area basis / frozen design), runs/<step>/inputs/<label>.pkl (the
session's loader values; imaging in imaging/<label>.pkl), fits.pkl (frozen fit, built on the desktop). No share reads.
Memory: {SBATCH_MEM} per task (desktop ~7 GB / worker); a task that runs out: resubmit that index with --mem=32G.

## 1. Transfer node  (ssh {user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b}
rsync -ah {share_b}/ {b}/
ls {b}/imaging | wc -l
```
(after a new step is added to the bundle, the same rsync copies only the new files)

## 2. Steps
""" + "\n".join(blocks) + """
## 3. After collect
collect prints the queue marker instruction; the COORDINATOR writes render_daqlk/logs/<step>.done (never this script).
"""
    (d / "COMMANDS.md").write_text(txt, encoding="utf-8", newline="\n")


SBATCH_MEM = "16G"


# ------------------------------------------------------------------------------------------------------- O2 task
_SHARE_READS: list = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if _is_ext(p):
            _SHARE_READS.append(p.replace("\\", "/"))


def run_task(b, step: str, label: str, fits=None, audit: bool = False) -> int:
    """O2 (and `check`): verify identity + masks, replay the session's loaders, `partial_refit.job` -> sessions json."""
    b = Path(b)
    meta = json.loads((b / "bundle.json").read_text())
    r = b / "runs" / step
    rmeta = json.loads((r / "run.json").read_text())
    verify_identity(rmeta["identity"])
    from wfield_local import mask_keys, o2_bundle
    inp = pickle.loads((r / "inputs" / f"{label}.pkl").read_bytes())
    img = pickle.loads((b / "imaging" / f"{label}.pkl").read_bytes())
    o2_bundle.check_inputs(meta, inp, label)
    o2_bundle.check_inputs(meta, img, label + " (imaging)")
    fits = Path(fits) if fits else r / "partial_refit" / "fits.pkl"
    sc = fits.parent / (fits.name + mask_keys.SIDECAR_SUFFIX)
    if not sc.exists() or sc.read_text(encoding="utf-8").strip() != meta[o2_bundle.MASK_FIELD]:
        raise o2_bundle.BundleMismatch(f"{fits}: mask sidecar missing or not the bundle's {meta[o2_bundle.MASK_FIELD]}")
    if (fits.parent / "sessions" / f"{label}.json").exists():
        print("done already", label)
        return 0
    if audit:
        sys.addaudithook(_audit)
    import partial_refit as pr
    t0 = time.time()
    with Replay({**img["memo"], **inp["memo"]}, inp["files"], inp["globs"]):
        fe = _fe()
        from wfield_local import config
        dd = design_digest(fe.session_design(label, fe._templates_of(str(r / "fit" / "templates.pkl"),
                                                                     config.animal_of(label))))
        if dd != inp["design"]:
            raise o2_bundle.BundleMismatch(f"{label}: design digest {dd} differs from the bundle's {inp['design']}")
        pr.job((label, str(r), str(fits)))
    if audit:
        print(f"share / cache reads during the task: {len(_SHARE_READS)} {sorted(set(_SHARE_READS))[:5]}")
        if _SHARE_READS:
            return 3
    print(f"{step} {label}: design {dd} ok; done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------- check
def _flatten(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flatten(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flatten(v, f"{pre}[{i}]")
    else:
        yield pre, x


def compare_sessions(j1: dict, j2: dict) -> tuple[float, list]:
    import math
    f1, f2 = dict(_flatten(j1)), dict(_flatten(j2))
    bad = sorted(set(f1) ^ set(f2))
    worst = 0.0
    for k in set(f1) & set(f2):
        a, c = f1[k], f2[k]
        if isinstance(a, float) or isinstance(c, float):
            if (math.isnan(a) if isinstance(a, float) else False) != (math.isnan(c) if isinstance(c, float) else False):
                bad.append(k)
            elif not (isinstance(a, float) and math.isnan(a)):
                worst = max(worst, abs(float(a) - float(c)))
        elif a != c:
            bad.append(k)
    return worst, bad


def cmd_check(a) -> int:
    """The bundled task (clean subprocess, ONLY env.json + the bundle's code, share reads audited) vs the desktop code
    path (`partial_refit.job` on the share with the same fits) for one label; outputs go under --out."""
    from wfield_local import mask_keys
    import partial_refit as pr
    d = _bundle_dir(a.name)
    r = d / "runs" / a.step
    out = Path(a.out)
    for sub in ("bundle", "desktop"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in ("fits.pkl", "fits.pkl" + mask_keys.SIDECAR_SUFFIX):
            shutil.copy2(r / "partial_refit" / f, out / sub / f)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_", "LOCANMF_",
                                                                    "COMBINED_ENCODING_", "OROFACIAL_", "FROZEN_"))}
    env.pop("PYTHONPATH", None)
    t0 = time.time()
    p = subprocess.run([sys.executable, str(d / "partial_runner.py"), a.step, "--label", a.label, "--fits",
                        str(out / "bundle" / "fits.pkl"), "--audit"], env=env, cwd=str(out), capture_output=True, text=True)
    print((p.stdout + p.stderr).strip()[-3000:])
    t1 = time.time()
    if p.returncode != 0:
        print("CHECK FAILED (bundled run)")
        return 1
    run = _run_dir(a.step)
    pr.job((a.label, str(run), str(out / "desktop" / "fits.pkl")))
    t2 = time.time()
    j1 = json.loads((out / "bundle" / "sessions" / f"{a.label}.json").read_text())
    j2 = json.loads((out / "desktop" / "sessions" / f"{a.label}.json").read_text())
    worst, bad = compare_sessions(j1, j2)
    n = sum(1 for _ in _flatten(j1))
    print(f"{a.label}: {n} values; max |bundle - desktop| = {worst:.3e}; differing keys / non-numeric: {bad[:5]} "
          f"({len(bad)}); bundled task {(t1 - t0) / 60:.1f} min, desktop {(t2 - t1) / 60:.1f} min")
    ok = not bad and worst < 1e-9
    print("CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------- collect
def cmd_collect(a) -> int:
    import datetime

    from wfield_local import mask_keys, o2_bundle, writeguard
    import partial_refit as pr
    d = _bundle_dir(a.name)
    meta = json.loads((d / "bundle.json").read_text())
    r = d / "runs" / a.step
    rmeta = json.loads((r / "run.json").read_text())
    try:
        o2_bundle.verify_masks(meta)
        verify_identity(rmeta["identity"])
    except o2_bundle.BundleMismatch as ex:
        raise SystemExit(f"refusing: this bundle is not of the current run ({ex})") from None
    run = _run_dir(a.step)
    if str(run) != rmeta["run_dir"]:
        raise SystemExit(f"run folder {run} differs from the bundle's {rmeta['run_dir']}")
    st = (run / "fit" / "pre_stats.pkl").stat()
    if st.st_size != rmeta["pre_stats"]["size"] or abs(st.st_mtime - rmeta["pre_stats"]["mtime"]) > 1:
        raise SystemExit(f"{run}/fit/pre_stats.pkl changed since the bundle was built (frozen run re-fit?)")
    labels = [x.strip() for x in (r / "labels.txt").read_text().splitlines() if x.strip()]
    src = r / "partial_refit" / "sessions"
    have = sorted(p.stem for p in src.glob("*.json"))
    missing = sorted(set(labels) - set(have))
    if missing and not a.partial:
        raise SystemExit(f"{len(missing)} of {len(labels)} sessions missing in {src}: {missing[:8]} (--partial to collect "
                         "anyway)")
    out = run / "partial_refit"
    dst = out / "sessions"
    writeguard.assert_writable(dst)
    dst.mkdir(parents=True, exist_ok=True)
    foreign = [f.name for f in dst.glob("*.json") if not (src / f.name).exists()
               or (src / f.name).read_bytes() != f.read_bytes()]
    if foreign:
        raise SystemExit(f"{dst} holds {len(foreign)} session file(s) not from this bundle ({foreign[:5]}): move them "
                         "aside first")
    fits = out / "fits.pkl"
    if fits.exists() and fits.read_bytes() != (r / "partial_refit" / "fits.pkl").read_bytes():
        raise SystemExit(f"{fits} exists and differs from the bundle's frozen fits: move it aside first")
    shutil.copy2(r / "partial_refit" / "fits.pkl", fits)
    mask_keys.write_sidecar(fits, meta[o2_bundle.MASK_FIELD])
    n = 0
    for f in sorted(src.glob("*.json")):
        s = f.read_text(encoding="utf-8")
        obj, end = json.JSONDecoder().raw_decode(s)        # a doubled task can leave a trailing byte (o2_encoding)
        (dst / f.name).write_text(json.dumps(obj) if s[end:].strip() else s, encoding="utf-8")
        n += 1
    (out / f"collected_o2_{a.name}.json").write_text(json.dumps(
        {"bundle": str(d), "step": a.step, "commit": meta["commit"], o2_bundle.MASK_FIELD: meta[o2_bundle.MASK_FIELD],
         "n_sessions": n, "missing": missing, "collected": datetime.datetime.now().isoformat(timespec="seconds")},
        indent=1), encoding="utf-8")
    print(f"collected {n} sessions -> {dst}; summarising (partial_refit.py --plot-only)", flush=True)
    pr.main(["--tag", STEPS[a.step][0], "--plot-only"])
    print(f"\nDONE. Queue bookkeeping is the coordinator's: when satisfied, create {QUEUE_LOGS}/{a.step}.done "
          f"(this script never writes it).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--step", required=True, choices=sorted(STEPS))
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True)
    b.add_argument("--jobs", type=int, default=2)
    b.add_argument("--mem", default=SBATCH_MEM)
    b.add_argument("--time", default="0-03:00:00")
    b.add_argument("--ignore-upstream", action="store_true", help="build although the upstream .done marker is absent")
    b.add_argument("labels", nargs="*")
    c = sub.add_parser("check")
    c.add_argument("--step", required=True, choices=sorted(STEPS))
    c.add_argument("--name", required=True)
    c.add_argument("--out", required=True, help="scratch folder for the two runs' outputs")
    c.add_argument("label")
    k = sub.add_parser("collect")
    k.add_argument("--step", required=True, choices=sorted(STEPS))
    k.add_argument("--name", required=True)
    k.add_argument("--partial", action="store_true", help="collect although some sessions are missing")
    a = ap.parse_args(argv)
    apply_step_env(a.step)                        # BEFORE wfield_local's env-reading modules are imported
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if a.cmd == "bundle":
        cmd_bundle(a)
        return 0
    return cmd_check(a) if a.cmd == "check" else cmd_collect(a)


if __name__ == "__main__":
    raise SystemExit(main())

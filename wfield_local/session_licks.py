"""THE SESSION LICK TABLE: one loader for "every video lick of a session", so every consumer reads the SAME lick set
(Priya 11-21, Q1 = (b): "the fused four-camera lick train defines the lick set" for 3-D kinematics, trial classes,
gates and lick-free windows; reconciled re-run plan section 2: "one shared loader").

TWO SOURCES, ONE TABLE SHAPE (`session_licks`):
  legacy  (fusion off)  cam4 `licks_all.csv` of `lp_licks.input_run()` (= `orofacial_cohort.RUN` unless the re-run
          routing env is set), every context -- what every consumer read before 11-22. cam1-only events are NOT in it;
          the consumers that add them (lp_licks, trial_classes, restnl cam4cam1, the quiet chain, the encoder's
          "combined" mode) keep doing so ONLY on this source (`fused_on()` False), byte-identical to before.
  fused   (fusion on)   the fused four-camera train (`scripts/camera_licks.py session --saved [--gapfix]`:
          `session_poses/camera_licks/<run>/<animal>_<date>_fused.csv`) through `camera_licks.lick_train` (D2 revised:
          any camera; guarded-only, unconfirmed side-only (`keep`) and long cam1-only licks out) and the VIDEO LICK FLOOR
          (`apply_floor`, 60 ms = `camera_licks.MIN_ILI_MS`, Priya 11-21: two video-lick peaks of one trial closer
          than that are one lick). cam1-only and side-only licks are already IN this train, so a consumer must not add
          cam1 events on top of it.
Columns (both): trial_id, position, lick_idx, on_ms, off_ms, t_ms (= the PEAK, ms from the cue, as licks_all),
context / onset_context (`orofacial_outputs.lick_context` / `onset_context`), table, daq_on_s, daq_peak_s,
``lick_key`` and ``source`` ("cam4" | "fused"); the fused table adds t_peak_ms, cams, n_views, seen_<cam>.

WHICH SOURCE (`fused_run`): env `WIDEFIELD_FUSED_LICKS` = a camera_licks folder name (e.g.
``lp_side_r2b_prod_20261118_3net_20261121_gapfix60``) or ``off``; unset -> `FUSED_DEFAULT` (None = legacy until the
adoption commit sets it; Priya 11-21 Q2). Spawned workers inherit the env. Every cache key that depends on the lick set
takes `signature()` ("" when off, so the default keys never move).

THE LICK KEY (`lick_key`, 11-22; reconciled plan: "lick_uid row-index join breaks silently"). ``lick_uid`` was the ROW
of the session's licks_all.csv, so any change to that table (v37, a re-run, the fused train) silently re-pointed every
joined kinematics row to another lick. The key is (trial_id, PEAK ms from the cue) at 0.1 ms: unique within a session
(no two licks of a trial share a peak; checked on 23 v37 sessions, 0 duplicates; the fused train is >= 60 ms apart),
and a fused lick that cam4 saw keeps cam4's peak (`camera_licks.PEAK_PRIORITY`), so a cam4 row and its fused lick
carry the same key. `rows_for` joins a kinematics table to a lick table by key and RAISES on any key it cannot find; a
pre-key table (no ``lick_key`` column) falls back to ``lick_uid`` only after checking every joined row's trial and peak.

PRECISION (11-22): the 11-21 fused / events CSVs were written with ``float_format="%.4g"``, which keeps 10 ms at
10-100 s from the cue (ITI licks) and makes ``daq_on_s`` useless after 1000 s. `scripts/camera_licks.py` now writes
``%.10g`` and a manifest (`MANIFEST`, ``format`` = `FORMAT_VERSION`); `session_licks` refuses a fused folder without
it (`FusedFormatError`) instead of reading rounded times. DAQ times are never taken from the fused CSVs: they are
recomputed from the cue-relative times through cam4's windows index (`orofacial_outputs.trial_ms_to_daq`, the same
function licks_all uses).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

FUSED_ENV = "WIDEFIELD_FUSED_LICKS"
#: the ADOPTED fused train folder (under session_poses/camera_licks); None = the legacy cam4 + cam1-only lick set. Set by
#: the adoption commit only (Priya 11-21 Q2: "one commit once the consumers are rewired and the lick_uid key join is in").
FUSED_DEFAULT: str | None = None
OFF_VALUES = ("off", "none", "legacy", "0")
#: fused-folder format: 2 = full-precision CSVs (``%.10g``) + `MANIFEST`. Folders without the manifest (11-21) are refused.
FORMAT_VERSION = 2
MANIFEST = "fused_manifest.json"
#: lick_key resolution (ms): the peak is rounded to this before it becomes part of the key
KEY_RES_MS = 0.1
CONTEXT_TABLE = {"post_cue": "licks", "pre_cue": "licks_precue", "excluded_0_70": "licks_excluded_0_70",
                 "iti": "licks_iti"}


class FusedFormatError(RuntimeError):
    """The fused folder predates the full-precision format (no `MANIFEST`): re-fuse it before use."""


class SessionLicksUnavailable(FileNotFoundError):
    """No lick table for the session in the selected source (no video, not fused, or no cam4 windows)."""


class LickJoinError(RuntimeError):
    """A kinematics row could not be matched to exactly one lick of the session's lick table."""


# --------------------------------------------------------------------------------------------- source selection
def fused_run() -> str | None:
    """The fused-train folder in force (`FUSED_ENV`, else `FUSED_DEFAULT` -- for COHORT 1 only: the fused train is a
    PS92-95 product and cohort 2 is never pooled with it), or None = legacy."""
    v = os.environ.get(FUSED_ENV)
    if v is None or not v.strip():
        if FUSED_DEFAULT is None:
            return None
        from wfield_local import config
        return FUSED_DEFAULT if config.cohort() == "cohort1" else None
    v = v.strip()
    if v.lower() in OFF_VALUES:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_]+", v):
        raise ValueError(f"{FUSED_ENV}={v!r}: expected a camera_licks folder name (letters / digits / '_') or 'off'")
    return v


def fused_on() -> bool:
    return fused_run() is not None


def fused_dir(rv=None, run: str | None = None) -> Path:
    """<dlc>/session_poses/camera_licks/<run> (`fused_run` when ``run`` is None; raises when the fusion is off)."""
    from wfield_local.paths import PathResolver
    run = run or fused_run()
    if run is None:
        raise SessionLicksUnavailable(f"no fused lick train selected ({FUSED_ENV} unset / off)")
    return Path((rv or PathResolver()).root("dlc")) / "session_poses" / "camera_licks" / run


def write_manifest(out: Path, **info) -> Path:
    """The fused folder's format manifest (written by `scripts/camera_licks.py` with every fused table)."""
    p = Path(out) / MANIFEST
    meta = {"format": FORMAT_VERSION, "float_format": "%.10g", **info}
    p.write_text(json.dumps(meta, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return p


def check_format(d: Path) -> dict:
    """The manifest of fused folder ``d``; `FusedFormatError` when it is missing or older than `FORMAT_VERSION`."""
    p = Path(d) / MANIFEST
    if not p.exists():
        raise FusedFormatError(f"{d}: no {MANIFEST} -- written before the full-precision format (11-22; its times are "
                               "rounded to 4 significant digits). Re-fuse: python -m scripts.camera_licks session ... "
                               "--saved --gapfix --out-tag <new>")
    meta = json.loads(p.read_text(encoding="utf-8"))
    if int(meta.get("format", 0)) < FORMAT_VERSION:
        raise FusedFormatError(f"{d}: format {meta.get('format')} < {FORMAT_VERSION}; re-fuse")
    return meta


_SIG: dict = {}


def signature(rv=None) -> str:
    """Cache-key part of the lick set: "" on the legacy source (default keys unchanged), else ``fused:<run>:<sha10>``
    over the fused folder's tables (names, sizes, mtimes) and the code of the train rules (`camera_licks.lick_train`,
    `apply_floor`). Memoised per process."""
    run = fused_run()
    if run is None:
        return ""
    import hashlib
    import inspect

    from wfield_local import camera_licks as CL
    d = fused_dir(rv, run)
    if str(d) not in _SIG:
        h = hashlib.sha1()
        for f in sorted(d.glob("*_fused.csv")) if d.exists() else []:
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
        m = d / MANIFEST
        if m.exists():
            h.update(m.read_bytes())
        h.update(inspect.getsource(CL.lick_train).encode())
        h.update(inspect.getsource(apply_floor).encode())
        h.update(f"floor{CL.MIN_ILI_MS}:long{CL.LONG_CAM1_ONLY_MS}".encode())
        _SIG[str(d)] = h.hexdigest()[:10]
    return f"fused:{run}:{_SIG[str(d)]}"


# --------------------------------------------------------------------------------------------- keys
def lick_key(trial_id, peak_ms) -> np.ndarray:
    """``"<trial_id>:<peak ms at 0.1 ms>"`` per lick (see the module docstring)."""
    tid = np.asarray(trial_id)
    pk = np.round(np.asarray(peak_ms, float) / KEY_RES_MS) * KEY_RES_MS
    return np.array([f"{int(t)}:{p:.1f}" for t, p in zip(tid, pk)], dtype=object)


def rows_for(K: pd.DataFrame, A: pd.DataFrame, label: str = "") -> np.ndarray:
    """Row positions in lick table ``A`` (one session) of every row of kinematics table ``K`` (same session).

    KEY JOIN (``K.lick_key``): every key must occur exactly once in ``A`` (`LickJoinError` otherwise) -- reordering or
    adding licks to ``A`` can never re-point a row. LEGACY (``K`` written before 11-22: ``lick_uid`` = row of
    licks_all.csv, ``t_peak_cam4`` = that row's peak at 4 significant digits): the row is used only if its trial_id
    matches and its peak agrees to the rounding of the stored value; any mismatch raises (the silent misjoin the key
    replaces)."""
    akey = A["lick_key"] if "lick_key" in A else pd.Series(lick_key(A.trial_id, A.t_ms), index=A.index)
    pos = pd.Series(np.arange(len(A)), index=akey.to_numpy())
    if pos.index.duplicated().any():
        dup = pos.index[pos.index.duplicated()][:3].tolist()
        raise LickJoinError(f"{label}: lick table has duplicate lick keys {dup}")
    if "lick_key" in K:
        r = pos.reindex(K.lick_key.to_numpy())
        if r.isna().any():
            miss = K.lick_key[r.isna().to_numpy()].head(3).tolist()
            raise LickJoinError(f"{label}: {int(r.isna().sum())} kinematics lick key(s) not in the lick table, e.g. "
                                f"{miss} -- the kinematics and the lick table come from different lick sets")
        return r.to_numpy(int)
    if "lick_uid" not in K:
        raise LickJoinError(f"{label}: kinematics table has neither lick_key nor lick_uid")
    uid = K.lick_uid.to_numpy(int)
    if len(uid) and (uid.min() < 0 or uid.max() >= len(A)):
        raise LickJoinError(f"{label}: lick_uid outside the lick table (rows {len(A)}): stale kinematics")
    ok = A.trial_id.to_numpy()[uid] == K.trial_id.to_numpy()
    if "t_peak_cam4" in K:
        a, k = A.t_ms.to_numpy(float)[uid], K.t_peak_cam4.to_numpy(float)
        tol = np.maximum(0.6, 5.1e-4 * np.abs(k))                # %.4g: half a unit in the 4th significant digit
        ok &= ~(np.abs(a - k) > tol)
    if not ok.all():
        raise LickJoinError(f"{label}: {int((~ok).sum())} / {len(ok)} lick_uid rows point at a different lick (trial / "
                            "peak mismatch) -- the lick table changed since the kinematics were written; rebuild them "
                            "(pose3d_kinematics writes lick_key since 11-22)")
    return uid


#: kinematics columns a key join needs (``lick_key``; ``lick_uid`` / ``trial_id`` / ``t_peak_cam4`` for pre-key tables)
KIN_JOIN_COLS = ("lick_key", "lick_uid", "trial_id", "t_peak_cam4")


def kin_by_row(K: pd.DataFrame, A: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """One session's kinematics ``K`` re-indexed by the ROW of lick table ``A`` each kinematics row belongs to
    (`rows_for`): the drop-in for the old ``K.drop_duplicates("lick_uid").set_index("lick_uid")``, which trusted that
    lick_uid still pointed at the same row of A."""
    K = K.drop_duplicates("lick_key" if "lick_key" in K else "lick_uid")
    return K.set_axis(rows_for(K, A, label), axis=0)


def align_kin(L: pd.DataFrame, K: pd.DataFrame, cols: list, label: str = "") -> pd.DataFrame:
    """``K[cols]`` (one session's 3-D kinematics rows) aligned to the rows of lick table ``L`` (trial_id, t_ms, lick_idx),
    NaN where a lick has no kinematics row (fewer than 3 3-D frames is a legitimate miss). By ``lick_key`` when ``K``
    has it -- a key cannot point at the wrong lick. A pre-key ``K`` joins on (trial_id, lick_idx) as before, but every
    matched row must agree on the peak (``t_peak_cam4`` vs ``L.t_ms``, to its 4-significant-digit rounding);
    `LickJoinError` otherwise (a kinematics table from another lick table / run)."""
    if "lick_key" in K:
        Kk = K.drop_duplicates("lick_key").set_index("lick_key")
        return Kk.reindex(lick_key(L.trial_id, L.t_ms))[cols].reset_index(drop=True)
    J = L[["trial_id", "lick_idx", "t_ms"]].reset_index(drop=True).merge(
        K.drop_duplicates(["trial_id", "lick_idx"]), on=["trial_id", "lick_idx"], how="left")
    if "t_peak_cam4" in J:
        a, k = J.t_ms.to_numpy(float), J.t_peak_cam4.to_numpy(float)
        bad = np.isfinite(k) & (np.abs(a - k) > np.maximum(0.6, 5.1e-4 * np.abs(k)))
        if bad.any():
            raise LickJoinError(f"{label}: {int(bad.sum())} kinematics rows joined on (trial_id, lick_idx) have another "
                                "lick's peak -- the kinematics were built on a different lick table; rebuild them")
    return J[cols]


# --------------------------------------------------------------------------------------------- the video-lick floor
def apply_floor(U: pd.DataFrame, min_ili_ms: float | None = None) -> pd.DataFrame:
    """Merge fused licks of one trial whose PEAKS are closer than ``min_ili_ms`` (default `camera_licks.MIN_ILI_MS`,
    60 ms, Priya 11-21: one protrusion split by a tracking dropout / a cut; `pose3d.lick_peaks` merges 3-D runs the same
    way). Consecutive licks chain (a lick < floor after the previous one joins its group). Merged lick: span = union,
    seen_<cam> = any, n_views / cams recomputed, peak = the peak of the member a higher-priority camera saw
    (`camera_licks.PEAK_PRIORITY`, ties -> the earlier); ``floor_merged`` = members merged into it (0 = untouched)."""
    from wfield_local import camera_licks as CL
    floor = CL.MIN_ILI_MS if min_ili_ms is None else float(min_ili_ms)
    if not len(U):
        return U.assign(floor_merged=np.zeros(0, int))
    U = U.sort_values(["trial_id", "t_peak_ms"], kind="mergesort").reset_index(drop=True)
    d = U.groupby("trial_id").t_peak_ms.diff()
    grp = (~(d < floor)).cumsum().to_numpy()
    if grp[-1] == len(U):
        return U.assign(floor_merged=0)
    seen_cols = [f"seen_{c}" for c in CL.CAMS if f"seen_{c}" in U]
    rank = np.full(len(U), len(CL.PEAK_PRIORITY), int)
    for i, c in reversed(list(enumerate(CL.PEAK_PRIORITY))):
        if f"seen_{c}" in U:
            rank = np.where(U[f"seen_{c}"].astype(bool).to_numpy(), i, rank)
    U = U.assign(_grp=grp, _rank=rank, _ord=np.arange(len(U)))
    head = U.sort_values(["_grp", "_rank", "_ord"], kind="mergesort").groupby("_grp").head(1).set_index("_grp")
    g = U.groupby("_grp")
    out = head.copy()
    out["on_ms"] = g.on_ms.min()
    out["off_ms"] = g.off_ms.max()
    for c in seen_cols:
        out[c] = g[c].any()
    if seen_cols:
        S = out[seen_cols].astype(bool)
        out["n_views"] = S.sum(1).to_numpy()
        names = [c[5:] for c in seen_cols]
        out["cams"] = ["+".join(sorted(n for n, v in zip(names, r) if v)) for r in S.to_numpy()]
    out["floor_merged"] = (g.size() - 1).reindex(out.index).to_numpy()
    out = out.reset_index(drop=True).drop(columns=["_rank", "_ord"])
    out = out.sort_values(["trial_id", "t_peak_ms"], kind="mergesort").reset_index(drop=True)
    if "lick_idx" in out:
        out["lick_idx"] = out.groupby("trial_id").cumcount()
    return out


# --------------------------------------------------------------------------------------------- the loaders
def _split(label: str) -> tuple[str, str]:
    from wfield_local import config
    an, d = label.split("_")[0], label.split("_")[1]
    return an, config.date8(d)


def _cam4_dir(rv, animal: str, date: str) -> Path:
    from wfield_local import lp_licks
    from wfield_local import orofacial_outputs as oo
    return oo.session_dir(rv, lp_licks.input_run(), animal, date)


def _windows_ix(sd: Path) -> pd.DataFrame:
    w = np.load(sd / "windows.npz", allow_pickle=False)
    return pd.DataFrame({k[3:]: w[k] for k in ("ix_trial_id", "ix_t_ms", "ix_daq_s", "ix_strobe_ms", "ix_stop_ms")})


def fused_table(U: pd.DataFrame, ix: pd.DataFrame) -> pd.DataFrame:
    """A fused train (`camera_licks.fuse` columns, already through `lick_train` / `apply_floor`) as the session lick
    table, given cam4's windows index ``ix`` (trial_id, t_ms, daq_s, strobe_ms, stop_ms). Pure (tested)."""
    from wfield_local import orofacial_outputs as oo
    A = U.copy()
    A["t_ms"] = A.t_peak_ms.to_numpy(float)
    tr = ix.groupby("trial_id").agg(strobe_ms=("strobe_ms", "first"), stop_ms=("stop_ms", "first"))
    t = tr.reindex(A.trial_id)
    A["context"] = oo.lick_context(A.t_ms, t.strobe_ms.to_numpy(float), t.stop_ms.to_numpy(float))
    A["onset_context"] = oo.onset_context(A.on_ms, t.strobe_ms.to_numpy(float), t.stop_ms.to_numpy(float))
    A["table"] = A.context.map(CONTEXT_TABLE).to_numpy()
    # ITI segment, as licks_all's ITI table: "lead" = before this trial's spout arrives (peak < strobe; the window
    # label is the UPCOMING trial), "tail" = after its stop; blank outside the ITI
    iti = A.context.to_numpy() == "iti"
    lead = A.t_ms.to_numpy(float) < t.strobe_ms.to_numpy(float)
    A["segment"] = np.where(iti, np.where(lead, "lead", "tail"), "")
    look = ix.drop_duplicates(["trial_id", "t_ms"]).set_index(["trial_id", "t_ms"]).daq_s
    A["daq_on_s"] = oo.trial_ms_to_daq(ix, A.trial_id, A.on_ms, look=look)
    A["daq_peak_s"] = oo.trial_ms_to_daq(ix, A.trial_id, A.t_ms, look=look)
    A = A[A.trial_id.isin(tr.index)]                       # a lick of a trial cam4's windows do not hold: no clock
    A = A.sort_values(["trial_id", "t_ms"], kind="mergesort")
    A["lick_idx"] = A.groupby("trial_id").cumcount()
    A["lick_key"] = lick_key(A.trial_id, A.t_ms)
    A["source"] = "fused"
    return A.sort_values("daq_on_s", kind="stable").reset_index(drop=True)


_MEMO: dict = {}


def session_licks(label: str, rv=None) -> pd.DataFrame:
    """THE session lick table of ``label`` ("PS92_0818" or "PS92_20260818") in the selected source (module
    docstring). `SessionLicksUnavailable` when the source has no table for the session. Memoised per process on the
    input files' identity; callers get a copy."""
    from wfield_local.paths import PathResolver
    rv = rv or PathResolver()
    an, d = _split(label)
    sd = _cam4_dir(rv, an, d)
    run = fused_run()
    if run is None:
        f = sd / "licks_all.csv"
        ins = [f]
    else:
        fd = fused_dir(rv, run)
        f = fd / f"{an}_{d}_fused.csv"
        ins = [f, sd / "windows.npz", fd / MANIFEST]
    if not all(p.exists() for p in ins):
        miss = [str(p) for p in ins if not p.exists()]
        if run is not None and f.exists() and not (fd / MANIFEST).exists():
            check_format(fd)                                  # raises FusedFormatError (old precision)
        raise SessionLicksUnavailable(f"{an}_{d}: no lick table ({'; '.join(miss)})")
    key = (run, str(f), tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in ins))
    if key not in _MEMO:
        if len(_MEMO) > 32:
            _MEMO.clear()
        if run is None:
            A = pd.read_csv(f)
            A["lick_key"] = lick_key(A.trial_id, A.t_ms)
            A["source"] = "cam4"
        else:
            from wfield_local import camera_licks as CL
            check_format(fd)
            U = CL.lick_train(pd.read_csv(f))
            A = fused_table(apply_floor(U), _windows_ix(sd))
        _MEMO[key] = A
    return _MEMO[key].copy()


def session_onsets_s(label: str, rv=None, contexts: tuple | None = None, col: str = "daq_on_s") -> np.ndarray:
    """Sorted finite DAQ times (s) of the session's licks (``col``: daq_on_s | daq_peak_s), optionally only
    ``contexts`` (`orofacial_outputs.CONTEXTS`)."""
    A = session_licks(label, rv)
    if contexts is not None:
        A = A[A.context.isin(contexts)]
    t = A[col].to_numpy(float)
    return np.sort(t[np.isfinite(t)])


def kin_all_name() -> str:
    """The whole-session 3-D kinematics folder of the selected lick set: ``kinematics_fused`` (fused train,
    `scripts/pose3d_kinematics --licks fused`) or ``kinematics_all`` (legacy cam4 licks_all, ``--licks all``)."""
    return "kinematics_fused" if fused_on() else "kinematics_all"


def describe() -> str:
    """One line for captions / logs: which lick set is in force."""
    run = fused_run()
    if run is None:
        return "licks: cam4 licks_all (+ cam1-only events where the consumer adds them)"
    return f"licks: fused four-camera train {run} (lick_train + {_floor_text()} video-lick floor)"


def _floor_text() -> str:
    from wfield_local import camera_licks as CL
    return f"{CL.MIN_ILI_MS:g} ms"

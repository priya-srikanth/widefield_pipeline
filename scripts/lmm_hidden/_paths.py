"""Overridable input / output locations for the LMM hidden-slide extractors (a8 11-17: re-render on the cohort-1 rebuild).

Every default stays the 11-03 v36 location (the published 11-03 figures re-run byte-identically with no env set). A
re-run on another tree sets the env vars below (or passes --root / --outdir); nothing is edited in place.

  LMM_HIDDEN_OUTDIR         output folder (render.OUTDIR; every extractor's --outdir default)
  LMM_HIDDEN_ROOT           default --root of the imaging_encoding-rooted extractors (decoder_retention_pooled,
                            fig9i3_cls_pooled, key_figure_BC, recovery_timing_pooled_epoch_lick, spec_groups)
  LMM_HIDDEN_ITI_ROOT       default --root of the iti_licks-rooted extractors (lick_*, curve_*)
  LMM_HIDDEN_DIR_<SUB>      one sub-folder of --root, by name: <SUB> = ATTEMPT_CODING, ATTEMPT_CODING_PP, ITI_LICKS
                            (a run-mode-tagged tree keeps each family in <family>/<run tag>/, not <root>/<family>/)
  LMM_HIDDEN_SESS_DIR       the iti_licks per-session extract folder (default <iti root>/sessions_roi8fr; on a tagged
                            tree it is content-keyed: iti_licks/<hemo/lnmf/pre parts>/sessions_roi8fr)
  LMM_HIDDEN_ITI_VARIANT    the ITI-set folder under the iti root (default itidock_main = the DOCKED-spout ITI set;
                            "." = the root itself, i.e. ALL ITI licks)
  LMM_HIDDEN_DOCKED_AS      when the ITI set is not the docked one: the suffix that replaces "_docked" in the output
                            names and the word that replaces "DOCKED" / "docked" in the titles (e.g. "alliti"), so an
                            all-ITI re-render can never be mistaken for (or overwrite) a docked figure
  LMM_HIDDEN_NAME_SUFFIX    appended to every output name (render.render)
  LMM_HIDDEN_ALLOW_NO_PUB   "1": a figure whose published comparison table is absent on this tree (e.g. the docked ITI
                            ramp tables were never re-run) is fitted WITHOUT the row-for-row reproduction check, and
                            says so loudly; default = refuse (SystemExit), as on 11-03
"""
from __future__ import annotations

import os
from pathlib import Path

V36_IE = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
V36_ITI = V36_IE / "iti_licks"


def env_path(var: str, default) -> Path:
    v = os.environ.get(var, "").strip()
    return Path(v) if v else Path(default)


def ie_root(default=V36_IE) -> Path:
    return env_path("LMM_HIDDEN_ROOT", default)


def iti_root(default=V36_ITI) -> Path:
    return env_path("LMM_HIDDEN_ITI_ROOT", default)


def sub(root, name: str) -> Path:
    """<root>/<name>, unless LMM_HIDDEN_DIR_<NAME> names the folder."""
    return env_path(f"LMM_HIDDEN_DIR_{name.upper()}", Path(root) / name)


def sess_dir(root, sess: str) -> Path:
    return env_path("LMM_HIDDEN_SESS_DIR", Path(root) / sess)


def variant(root, default: str = "itidock_main") -> Path:
    v = os.environ.get("LMM_HIDDEN_ITI_VARIANT", default).strip() or default
    return Path(root) if v == "." else Path(root) / v


def docked_as() -> str:
    return os.environ.get("LMM_HIDDEN_DOCKED_AS", "").strip()


def out_name(name: str) -> str:
    """Output name: "_docked" -> "_<LMM_HIDDEN_DOCKED_AS>" when set, then + LMM_HIDDEN_NAME_SUFFIX (a re-render that must
    not overwrite an earlier one in the same folder)."""
    d = docked_as()
    name = name.replace("_docked", f"_{d}") if d else name
    return name + os.environ.get("LMM_HIDDEN_NAME_SUFFIX", "").strip()


def out_title(title: str) -> str:
    d = docked_as()
    if not d:
        return title
    word = {"alliti": "ALL ITI licks (not docked)"}.get(d, d)
    return title.replace("DOCKED", word).replace("docked", word)


def allow_no_pub() -> bool:
    return os.environ.get("LMM_HIDDEN_ALLOW_NO_PUB", "") == "1"


def hit_gate(S, value: str, n_col: str, by=("animal", "epoch"), return_rows: bool = False):
    """Rows of ``S`` with a finite ``value``, minus every animal x epoch cell whose sessions sum to < analysis_kit.MIN_HITS
    hits in ``n_col`` -- the code-on-hits gate of the main figures (Priya 11-07: recovery_timing._hit_gated,
    key_figure._gated, decoder_retention_sessions._cell). Applied when ``n_col`` exists (the v36 tables predate it).
    Returns (kept rows, list of dropped cells) [, dropped rows when ``return_rows`` -- Priya 11-17: drawn as "<20 hits",
    shown but not tested]."""
    import numpy as np
    from wfield_local import analysis_kit as ak
    S = S[np.isfinite(S[value].to_numpy(float))]
    if n_col not in S:
        return (S, [], S.iloc[0:0]) if return_rows else (S, [])
    keep, dropped, rows = [], [], []
    for k, g in S.groupby(list(by), sort=False):
        if ak.hits_ok(g[n_col].to_numpy(float)):
            keep.append(g)
        else:
            dropped.append(f"{'/'.join(map(str, k))} ({int(np.nansum(g[n_col].to_numpy(float)))} hits)")
            rows.append(g)
    pd = __import__("pandas")
    kept = S.iloc[0:0] if not keep else pd.concat(keep)
    if return_rows:
        return kept, dropped, (S.iloc[0:0] if not rows else pd.concat(rows))
    return kept, dropped

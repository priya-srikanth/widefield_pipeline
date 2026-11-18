"""BEHAVIORAL STATES after the stroke: the shared readouts of the GLM-HMM ENGAGEMENT states, the AR-HMM MOVEMENT
SYLLABLES and the treadmill PUSH (Priya 11-17). One collector pass over the sessions builds every per-session and
per-trial table; the results page (`scripts/states_page_figs.py`) and any later analysis read them from here.

    python -m wfield_local.behavior_states collect [--jobs 8]      # -> out_dir(): csv.gz tables + traces.npz

EVERY CHOICE IS MADE ONCE, HERE OR UPSTREAM (Priya 11-17: "if we change one thing in the future it will seamlessly
deploy to downstream analyses"):
  * models ............ GLM-HMM K / design = GLMHMM_K / GLMHMM_DESIGN (tables written by `glm_hmm`); AR-HMM fit =
                         `arhmm_states.REFERENCE_TAG`, read from `arhmm_states.fits_dir()`; features from
                         `arhmm_features.out_dir()`.
  * state ROLES ........ `glmhmm_roles`: from each state's own posterior-weighted profile (engaged = attempt rate >=
                         ENGAGED_ATTEMPT; disengaged = the highest no-response rate of the rest; partial = the others),
                         never hard-coded indices, so a refit or a new K renames nothing downstream.
  * syllable GROUPS .... `state_groups`: the ``group`` column of the REVIEWED ``state_labels.csv`` beside the model's
                         reels (`arhmm_reels.reels_dir`). Regroup there; every readout follows.
  * epochs / baseline .. `epochs.epoch_of` + `config.early_pre` (August-only pre-stroke baseline).
  * spouts ............. `orofacial_cohort.pos_key` (ipsi / contra from animals.yaml stroke_laterality), order
                         `orofacial_cohort.POS_ORDER`, names `orofacial_cohort.pos_disp`.
  * thresholds ......... walking = configs/defaults.yaml segmentation.running.thresh_speed_mm_s (the DAQ running
                         definition); push = segmentation.push.* (backward speed, cue and control windows).
  * gates .............. LICK working period = `trial_classes.camera_engaged_period` (first .. last camera lick: a
                         PERIOD, so it does not select on the trial's own response); GLM-HMM engaged = argmax state has
                         the "engaged" role.
  * trial cycle ........ previous trial's stop .. this trial's stop (ITI + pre-cue + response), from the camera trial
                         table (`trial_classes.session_trials`: cue_daq + stop_ms).
  * statistics ......... `orofacial_cohort.epoch_tables` (per-animal epoch table + paired animal-weighted
                         `analysis_kit.boot_delta` vs each animal's OWN pre); levels with `analysis_kit.boot_ci`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GLMHMM_K = 5                      # Priya 11-17: "let's use k=5" (held-out selection: select_k_pooled_all_v2.csv)
GLMHMM_DESIGN = "v2"              # shared animal offsets, no previous-outcome input (handoff 11-17)
ENGAGED_ATTEMPT = 0.9             # a GLM-HMM state is ENGAGED when >= 90 % of its trials are attempts (contact /
                                  #   incomplete); K5 v2: engaged states .99, partial .51, disengaged .04
LICK_GROUPS = ("lick_ipsi", "lick_central", "lick_contra")
#: DISPLAY order of the reviewed syllable groups (figures, tables); a group missing here is appended in label order,
#: so renaming / adding a group in state_labels.csv never breaks a readout
GROUP_DISPLAY = ("quiet", "nose_slow", "jaw_chew", "lick_ipsi", "lick_central", "lick_contra", "push", "walk_slow",
                 "walk")
CUE_ALIGN_S = (-2.0, 4.5)         # cue-aligned traces (Priya 11-17: "-2 to +4.5")
JAW_HALF_S = 0.15                 # syllables at the jaw onset: +- this
RESP_MAX_S = 2.0                  # response window for the per-trial syllable composition: cue .. cue + min(stop, 2 s)
ROLES = ("engaged", "partial", "disengaged")
WHEEL_CHANNELS = ("fwd", "back", "p_walk", "p_push")


# ------------------------------------------------------------------------------------------------ shared choices

def session_epoch(label: str) -> str | None:
    """The session's epoch, or None for a session outside the analysis (no epoch, or pre-stroke before the cohort's
    August-only baseline cut)."""
    from wfield_local import config
    from wfield_local import epochs
    e = epochs.epoch_of(label)
    return None if e is None or (e == "pre" and config.early_pre(label)) else e


def _seg() -> dict:
    from wfield_local import config
    return config.defaults().get("segmentation") or {}


def walk_threshold() -> float:
    """Walking = forward treadmill speed above the DAQ running threshold (mm/s)."""
    return float(_seg()["running"]["thresh_speed_mm_s"])


def push_params() -> dict:
    """{'thresh': mm/s backward, 'cue': (a, b) s, 'control': (a, b) s} from configs/defaults.yaml segmentation.push."""
    p = _seg()["push"]
    return {"thresh": float(p["thresh_back_mm_s"]), "cue": tuple(float(x) for x in p["cue_window_s"]),
            "control": tuple(float(x) for x in p["control_window_s"])}


def laterality() -> dict:
    from wfield_local import config
    return {a: str((v or {}).get("stroke_laterality", "L")) for a, v in config.animals().items()}


def out_dir(tag: str | None = None) -> Path:
    """``LP_analysis/<run>/arhmm/v1/poststroke_<tag>``: the collector's tables and the page figures."""
    from wfield_local import arhmm_states as st
    return st.fits_dir().parent / f"poststroke_{tag or st.REFERENCE_TAG}"


# ------------------------------------------------------------------------------------------------ GLM-HMM

def glmhmm_table(K: int = GLMHMM_K, design: str = GLMHMM_DESIGN) -> pd.DataFrame:
    """The per-trial posterior table written by `glm_hmm --pooled-all` (all trials, all epochs), restricted to the
    analysis sessions, with animal / epoch / pos (ipsi-contra key) columns added."""
    from wfield_local import glm_hmm
    from wfield_local import orofacial_cohort as oc
    sfx = "" if design == "v1" else f"_{design}"
    T = pd.read_csv(Path(glm_hmm.out_dir()) / f"states_pooled_all_K{K}{sfx}.csv.gz")
    T = T[T.cls != "unknown"].copy()
    T["epoch"] = T.label.map(session_epoch)
    T = T[T.epoch.notna()].copy()
    T["animal"] = T.label.str.split("_").str[0]
    lat = laterality()
    T["pos"] = [oc.pos_key(p, lat.get(a, "L")) for p, a in zip(T.position, T.animal)]
    return T


def _pcols(T: pd.DataFrame) -> list[str]:
    return sorted((c for c in T if c.startswith("p_state")), key=lambda c: int(c[7:]))


def glmhmm_profile(T: pd.DataFrame) -> pd.DataFrame:
    """Per state: posterior-weighted attempt / no-response / jaw-only rates, mean running fraction, occupancy."""
    att = T.cls.isin(["contact", "incomplete"]).to_numpy(float)
    nr = T.cls.eq("no_response").to_numpy(float)
    jo = T.cls.eq("jaw_only").to_numpy(float)
    run = ((T.run_frac_precue.fillna(0) + T.run_frac_iti.fillna(0)) / 2).to_numpy(float)
    rows = []
    for c in _pcols(T):
        w = T[c].to_numpy(float)
        rows.append({"state": int(c[7:]), "attempt": np.average(att, weights=w), "no_response": np.average(nr, weights=w),
                     "jaw_only": np.average(jo, weights=w), "run": np.average(run, weights=w), "occupancy": w.mean()})
    return pd.DataFrame(rows).set_index("state")


def glmhmm_roles(T: pd.DataFrame) -> dict:
    """{state: 'engaged' | 'partial' | 'disengaged'} from the states' own profiles (module docstring)."""
    P = glmhmm_profile(T)
    eng = P.index[P.attempt >= ENGAGED_ATTEMPT].tolist()
    rest = P.drop(index=eng)
    dis = [int(rest.no_response.idxmax())] if len(rest) else []
    return {int(k): ("engaged" if k in eng else "disengaged" if k in dis else "partial") for k in P.index}


def glmhmm_engaged(T: pd.DataFrame) -> dict:
    """{label: set(trial_id)} of trials whose most probable state has the engaged role."""
    roles = glmhmm_roles(T)
    pc = _pcols(T)
    arg = np.array([int(pc[i][7:]) for i in T[pc].to_numpy().argmax(1)])
    eng = np.array([roles[k] == "engaged" for k in arg])
    return {lab: set(g.trial_id) for lab, g in T[eng].groupby("label")}


#: GLM-HMM STATE colors (Priya 11-18: state colors were too close to the spout colors). Hues kept clear of the spout
#: hues (`spout_behavior.SIDE_HUE` blue / purple / red) and the epoch colors (`orofacial_cohort.EPOCH_LINE` black /
#: magenta / gold / teal); engaged states = one green ramp (dark -> light in state order), separable by lightness under
#: color-vision deficiency; partial = orange; disengaged = gray.
ROLE_COLORS = {"engaged": ("#00441B", "#41AB5D", "#A1D99B"), "partial": ("#F28E2B", "#B35806"),
               "disengaged": ("#7F7F7F", "#BDBDBD")}


def glmhmm_state_colors(roles: dict) -> dict:
    """{state: color} from the roles (`ROLE_COLORS`): the n-th state of a role takes that role's n-th shade."""
    out, n = {}, {r: 0 for r in ROLE_COLORS}
    for k in sorted(roles):
        r = roles[k]
        shades = ROLE_COLORS[r]
        out[k] = shades[min(n[r], len(shades) - 1)]
        n[r] += 1
    return out


def glmhmm_params(K: int = GLMHMM_K, design: str = GLMHMM_DESIGN) -> dict | None:
    """The fitted pooled GLM-HMM (`glm_hmm --pooled-all --k K --params-only`: W, B, A, pi, X_mean, coding), in the state
    numbering of `glmhmm_table`; None if not written."""
    import json
    from wfield_local import glm_hmm
    sfx = "" if design == "v1" else f"_{design}"
    f = Path(glm_hmm.out_dir()) / f"params_pooled_all_K{K}{sfx}.json"
    return json.loads(f.read_text()) if f.exists() else None


def glmhmm_select_k(design: str = GLMHMM_DESIGN) -> pd.DataFrame:
    """Held-out K selection (`glm_hmm --pooled-all --select-k`), every K run (the K1-5 and K6-8 runs share folds)."""
    from wfield_local import glm_hmm
    sfx = "" if design == "v1" else f"_{design}"
    fs = sorted(Path(glm_hmm.out_dir()).glob(f"select_k_pooled_all{sfx}*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["K", "fold"])


def glmhmm_emission(params: dict, pos: str | None = None) -> dict:
    """Model emission probabilities per state: {output: array [K, n_classes]} at the average input (shared animal
    offsets at their mean), for spout ``pos`` (a `glm_hmm.POSITIONS` name) or averaged over the input mean."""
    ns = params["n_shared"]
    xm = np.asarray(params["X_mean"], float)
    x = xm[ns:].copy()
    if pos is not None:
        P = params["positions"]
        x[1:len(P)] = 0.0
        i = P.index(pos)
        if i > 0:
            x[i] = 1.0
    out = {}
    for j, name in enumerate(params["outputs"]):
        W = np.asarray(params["W"][j], float)               # [K, C-1, D]
        off = (np.asarray(params["B"][j], float) @ xm[:ns]) if ns else 0.0
        eta = np.concatenate([np.zeros((W.shape[0], 1)), W @ x + off], axis=1)
        eta -= eta.max(1, keepdims=True)
        pr = np.exp(eta)
        out[name] = pr / pr.sum(1, keepdims=True)
    return out


def occupancy_per_session(T: pd.DataFrame) -> pd.DataFrame:
    """Mean posterior per state and per role, one row per session (label, animal, epoch)."""
    roles = glmhmm_roles(T)
    pc = _pcols(T)
    S = T.groupby(["label", "animal", "epoch"])[pc].mean().reset_index()
    for r in ROLES:
        S[r] = S[[f"p_state{k}" for k, v in roles.items() if v == r]].sum(1)
    return S


# ------------------------------------------------------------------------------------------------ AR-HMM

def state_groups(tag: str | None = None) -> dict:
    """{group: [states]} from the reviewed ``state_labels.csv`` (``group`` column) beside the model's reels."""
    from wfield_local import arhmm_reels
    from wfield_local import arhmm_states as st
    L = pd.read_csv(arhmm_reels.reels_dir(tag or st.REFERENCE_TAG) / "state_labels.csv")
    L = L[L.group.notna()]
    G = {g: sorted(int(s) for s in d.state) for g, d in L.groupby("group", sort=False)}
    order = [g for g in GROUP_DISPLAY if g in G] + [g for g in G if g not in GROUP_DISPLAY]
    return {g: G[g] for g in order}


def _group_lut(groups: dict, n: int = 64) -> np.ndarray:
    lut = np.full(n, -1)
    for i, ks in enumerate(groups.values()):
        lut[ks] = i
    return lut


def load_session(label: str, tag: str | None = None):
    """(t_daq_s, run_fwd, run_back, z) for one session: features from `arhmm_features.out_dir()`, states from
    `arhmm_states.fits_dir()`."""
    from wfield_local import arhmm_features as af
    from wfield_local import arhmm_states as st
    F = np.load(Path(af.out_dir()) / "sessions" / f"{label}.npz", allow_pickle=True)
    cols = [str(c) for c in F["features"]]
    z = np.load(st.fits_dir() / f"states_{tag or st.REFERENCE_TAG}" / f"{label}.npz")["z"].astype(int)
    return F["t_daq_s"], F["X"][:, cols.index("run_fwd")], F["X"][:, cols.index("run_back")], z


#: features shown for each syllable group's motif (`arhmm_features` columns; tongue_ml / jaw_x / nose_x are signed
#: + = contralesional); treadmill shown as signed velocity (forward +, backward push -)
MOTIF_FEATURES = ("tongue_ext", "tongue_ml", "jaw_y", "nose_y", "run")
MOTIF_WIN_S = (-0.4, 0.8)


def motif_traces(item) -> dict | None:
    """Onset-aligned feature traces of each syllable GROUP in one session: {group: [n_onsets, n_bins, n_features]}
    means, plus onset counts and bout durations. ``item`` = (label, groups, tag). Bout = a run of the same group."""
    from wfield_local import arhmm_features as af
    from wfield_local import arhmm_states as st
    label, groups, tag = item
    F = np.load(Path(af.out_dir()) / "sessions" / f"{label}.npz", allow_pickle=True)
    cols = [str(c) for c in F["features"]]
    X = F["X"]
    z = np.load(st.fits_dir() / f"states_{tag or st.REFERENCE_TAG}" / f"{label}.npz")["z"].astype(int)
    if len(z) != len(X):
        return None
    lut = _group_lut(groups)
    g = np.where(z >= 0, lut[np.maximum(z, 0)], -1)
    feats = np.stack([X[:, cols.index(c)] if c != "run" else X[:, cols.index("run_fwd")] - X[:, cols.index("run_back")]
                      for c in MOTIF_FEATURES], axis=1)
    b0, b1 = (int(round(w * 25)) for w in MOTIF_WIN_S)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    stops = np.r_[starts[1:], len(g)]
    out = {}
    for gi, name in enumerate(groups):
        on = starts[g[starts] == gi]
        dur = (stops - starts)[g[starts] == gi] / 25.0
        on = on[(on + b0 >= 0) & (on + b1 < len(g))]
        if not len(on):
            continue
        seg = np.stack([feats[o + b0:o + b1 + 1] for o in on])
        out[name] = {"mean": np.nanmean(seg, 0), "n": len(on), "dur_med": float(np.median(dur)),
                     "frac": float(np.mean(g == gi))}
    return {"label": label, "groups": out}


# ------------------------------------------------------------------------------------------------ collector

def _window(t, x, a, b):
    return x[np.searchsorted(t, a):np.searchsorted(t, b)]


def session_readouts(item):
    """One session (module-level for the process pool): per-trial rows, per-session usage / transitions, and the
    cue-aligned traces. ``item`` = (label, glm_engaged_ids, groups, tag)."""
    from wfield_local import trial_classes as tc
    label, eng_ids, groups, tag = item
    T = tc.session_trials(label)
    if T is None or not len(T):
        return None
    T = T.sort_values("trial_id").reset_index(drop=True)
    t, fwd, back, z = load_session(label, tag)
    fs = 1.0 / np.median(np.diff(t))
    gn = list(groups)
    g = np.where(z >= 0, _group_lut(groups)[np.maximum(z, 0)], -1)
    pp, walk = push_params(), walk_threshold()
    period = tc.camera_engaged_period(label, T.cue_daq.to_numpy(float))
    stop = T.cue_daq.to_numpy(float) + np.nan_to_num(T.stop_ms.to_numpy(float) / 1000.0, nan=RESP_MAX_S)
    nb = int(round((CUE_ALIGN_S[1] - CUE_ALIGN_S[0]) * fs))
    keys = ("lick_in", "lick_out", "glm_in", "glm_out")
    wheel = {k: np.zeros((len(WHEEL_CHANNELS), nb)) for k in keys}
    nwheel = dict.fromkeys(keys, 0)
    etho, neth = np.zeros((len(gn), nb)), 0
    cyc = np.zeros(len(t), bool)
    rows = []
    for i, r in T.iterrows():
        c = float(r.cue_daq)
        engaged = r.trial_id in eng_ids
        if engaged:
            a = stop[i - 1] if i > 0 else c - 5.0
            cyc[np.searchsorted(t, a):np.searchsorted(t, stop[i])] = True
        bc, bk = _window(t, back, c + pp["cue"][0], c + pp["cue"][1]), _window(t, back, c + pp["control"][0],
                                                                                  c + pp["control"][1])
        fc, fk = _window(t, fwd, c + pp["cue"][0], c + pp["cue"][1]), _window(t, fwd, c + pp["control"][0],
                                                                                c + pp["control"][1])
        gr = _window(t, g, c, c + min(stop[i] - c, RESP_MAX_S))
        gr = gr[gr >= 0]
        row = {"label": label, "trial_id": r.trial_id, "position": r.position, "cls": r.cls,
               "glm_engaged": engaged, "lick_period": bool(period[i])}
        if len(bc) >= 0.5 * fs and len(bk) >= 0.5 * fs and np.isfinite(bc).all() and np.isfinite(bk).all():
            above = np.flatnonzero(bc > pp["thresh"])
            row.update(push=bool(len(above)), push_control=bool((bk > pp["thresh"]).any()), push_peak=float(bc.max()),
                       push_latency_s=pp["cue"][0] + above[0] / fs if len(above) else np.nan,
                       walk_cue=float(np.mean(fc > walk)), walk_control=float(np.mean(fk > walk)))
        for k, name in enumerate(gn):                                 # syllable composition of the response window
            row[f"resp:{name}"] = float(np.mean(gr == k)) if len(gr) else np.nan
        if np.isfinite(r.jaw_onset_ms):                               # syllables at the jaw onset
            j = c + r.jaw_onset_ms / 1000.0
            gj = _window(t, g, j - JAW_HALF_S, j + JAW_HALF_S)
            gj = gj[gj >= 0]
            for k, name in enumerate(gn):
                row[f"jaw:{name}"] = float(np.mean(gj == k)) if len(gj) else np.nan
        rows.append(row)
        j0 = np.searchsorted(t, c + CUE_ALIGN_S[0])
        sb, sf, sg = back[j0:j0 + nb], fwd[j0:j0 + nb], g[j0:j0 + nb]
        if len(sb) == nb and np.isfinite(sb).all() and np.isfinite(sf).all():
            x = np.stack([sf, sb, sf > walk, sb > pp["thresh"]])
            for k in ("lick_in" if period[i] else "lick_out", "glm_in" if engaged else "glm_out"):
                wheel[k] += x
                nwheel[k] += 1
        if engaged and len(sg) == nb and (sg >= 0).mean() > 0.9:
            etho[sg[sg >= 0], np.flatnonzero(sg >= 0)] += 1
            neth += 1
    use = {"label": label}
    lick = [gn.index(n) for n in LICK_GROUPS if n in gn]
    for name, m in (("all", g >= 0), ("engaged", cyc & (g >= 0))):
        gg = g[m]
        for k, n in enumerate(gn):
            use[f"{name}:{n}"] = float(np.mean(gg == k)) if len(gg) else np.nan
        lk = np.isin(gg, lick)
        for k in lick:
            use[f"{name}:share_{gn[k]}"] = float(np.mean(gg[lk] == k)) if lk.sum() else np.nan
    C = np.zeros((len(gn), len(gn)))                                  # transitions on engaged cycles, runs collapsed
    gm = np.where(cyc, g, -1)
    runs = [r_[0] for r_ in np.split(gm, np.flatnonzero(np.diff(gm) != 0) + 1)]
    for x_, y_ in zip(runs[:-1], runs[1:]):
        if x_ >= 0 and y_ >= 0:
            C[x_, y_] += 1
    return {"trials": pd.DataFrame(rows), "usage": use, "transitions": C,
            "wheel": {k: (wheel[k] / nwheel[k] if nwheel[k] >= 5 else None) for k in keys},
            "ethogram": etho / neth if neth >= 10 else None, "fs": fs}


def transition_table(res: dict, groups: dict, min_out: int = 20) -> pd.DataFrame:
    """P(next group | current group) per session (rows with < min_out transitions out -> NaN) + transition entropy."""
    gn = list(groups)
    rows = []
    for lab, r in res.items():
        C = r["transitions"]
        P = C / np.maximum(C.sum(1, keepdims=True), 1)
        row = {"label": lab}
        for i, a in enumerate(gn):
            for j, b in enumerate(gn):
                if i != j:
                    row[f"{a}>{b}"] = P[i, j] if C[i].sum() >= min_out else np.nan
        p = C.sum(1) / max(C.sum(), 1)
        row["entropy_bits"] = float(sum(p[i] * -(q[q > 0] * np.log2(q[q > 0])).sum() for i, q in enumerate(P)))
        rows.append(row)
    return pd.DataFrame(rows)


def collect(tag: str | None = None, jobs: int | None = None) -> Path:
    """Run `session_readouts` over every analysis session; write the tables + traces to `out_dir`."""
    from wfield_local import analysis_kit as ak
    from wfield_local import arhmm_states as st
    from wfield_local.writeguard import assert_writable
    tag = tag or st.REFERENCE_TAG
    T = glmhmm_table()
    eng = glmhmm_engaged(T)
    groups = state_groups(tag)
    labs = sorted(set(T.label) & {p.stem for p in (st.fits_dir() / f"states_{tag}").glob("*.npz")})
    res, fail = ak.fan_sessions([(lab, eng.get(lab, set()), groups, tag) for lab in labs], session_readouts,
                                jobs=jobs, key=lambda it: it[0])
    for f in fail:
        print("  failed:", f)
    res = {it[0]: r for it, r in res if r is not None}
    od = out_dir(tag)
    assert_writable(od)
    od.mkdir(parents=True, exist_ok=True)
    meta = lambda D: D.assign(animal=D.label.str.split("_").str[0], epoch=D.label.map(session_epoch))  # noqa: E731
    meta(pd.concat([r["trials"] for r in res.values()])).to_csv(od / "trials.csv.gz", index=False)
    meta(pd.DataFrame([r["usage"] for r in res.values()])).to_csv(od / "usage.csv.gz", index=False)
    meta(transition_table(res, groups)).to_csv(od / "transitions.csv.gz", index=False)
    meta(occupancy_per_session(T)).to_csv(od / "glmhmm_occupancy.csv.gz", index=False)
    arr = {}
    for lab, r in res.items():
        for k, v in r["wheel"].items():
            if v is not None:
                arr[f"wheel|{k}|{lab}"] = v
        if r["ethogram"] is not None:
            arr[f"etho|{lab}"] = r["ethogram"]
    fs = float(np.median([r["fs"] for r in res.values()]))
    np.savez_compressed(od / "traces.npz", **arr)
    (od / "collect.json").write_text(json.dumps({
        "tag": tag, "glmhmm": {"K": GLMHMM_K, "design": GLMHMM_DESIGN, "roles": glmhmm_roles(T)},
        "groups": groups, "push": push_params(), "walk_mm_s": walk_threshold(), "cue_align_s": CUE_ALIGN_S,
        "fs": fs, "n_sessions": len(res), "failed": [str(f) for f in fail]}, indent=1, default=str))
    print(f"{len(res)} sessions -> {od}")
    return od


def load_collected(tag: str | None = None) -> dict:
    """The collector's outputs: {'trials', 'usage', 'transitions', 'glmhmm_occupancy': DataFrame, 'traces': npz,
    'meta': dict}."""
    od = out_dir(tag)
    out = {k: pd.read_csv(od / f"{k}.csv.gz") for k in ("trials", "usage", "transitions", "glmhmm_occupancy")}
    out["traces"] = dict(np.load(od / "traces.npz"))
    out["meta"] = json.loads((od / "collect.json").read_text())
    return out


def per_session(trials: pd.DataFrame, gate: str | None = None) -> pd.DataFrame:
    """Per-session push / walking readouts from the trial table; ``gate`` = None (all trials), 'lick_period',
    '~lick_period', 'glm_engaged' or '~glm_engaged'."""
    D = trials
    if gate:
        col = gate.lstrip("~")
        D = D[~D[col]] if gate.startswith("~") else D[D[col]]
    g = D.groupby(["label", "animal", "epoch"])
    S = pd.DataFrame({"push_rate": g.push.mean(), "push_control_rate": g.push_control.mean(),
                      "push_peak": g.apply(lambda x: x.push_peak[x.push.astype(bool)].mean(), include_groups=False),
                      "push_latency_s": g.apply(lambda x: x.push_latency_s[x.push.astype(bool)].median(),
                                                include_groups=False),
                      "walk_cue": g.walk_cue.mean(), "walk_control": g.walk_control.mean()})
    S["push_cue_minus_control"] = S.push_rate - S.push_control_rate
    return S.reset_index()


def syllable_stats(tag: str, ref_tag: str | None = None, min_usage: float = 0.005) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session statistics of ANY decoded AR-HMM fit (e.g. the FROZEN pre-stroke model ``pre_s1`` applied to every
    session): usage and median dwell (s) per state, transition entropy (runs collapsed) and the share of frames in
    states rare before the stroke. Returns (per-session table, state names): each state is named by the reviewed
    group of the reference fit (`state_groups`, default `arhmm_states.REFERENCE_TAG`) it overlaps most, frame by frame,
    on the same sessions. Pre-stroke days are IN-SAMPLE for a model fit on them (usage, not likelihood, so the bias is
    small; the held-out folds serve the likelihood)."""
    from wfield_local import arhmm_model as am
    from wfield_local import arhmm_states as st
    ref_tag = ref_tag or st.REFERENCE_TAG
    groups = state_groups(ref_tag)
    lut = _group_lut(groups)
    gn = list(groups)
    d, dr = st.fits_dir() / f"states_{tag}", st.fits_dir() / f"states_{ref_tag}"
    labs = sorted(p.stem for p in d.glob("*.npz") if session_epoch(p.stem))
    Z = {lab: np.load(d / f"{lab}.npz")["z"].astype(int) for lab in labs}
    K = max(int(z.max()) for z in Z.values()) + 1
    over = np.zeros((K, len(gn)))
    for lab in labs:
        if (dr / f"{lab}.npz").exists():
            z, zr = Z[lab], np.load(dr / f"{lab}.npz")["z"].astype(int)
            ok = (z >= 0) & (zr >= 0)
            g = lut[zr[ok]]
            np.add.at(over, (z[ok][g >= 0], g[g >= 0]), 1)
    allz = np.concatenate([z[z >= 0] for z in Z.values()])
    use_all = np.bincount(allz, minlength=K) / len(allz)
    live = [k for k in range(K) if use_all[k] >= min_usage]
    names = pd.DataFrame({"state": live, "group": [gn[int(over[k].argmax())] if over[k].sum() else "?" for k in live],
                          "group_share": [over[k].max() / max(over[k].sum(), 1) for k in live],
                          "usage_all": [use_all[k] for k in live]})
    pre_use = np.bincount(np.concatenate([Z[lab][Z[lab] >= 0] for lab in labs if session_epoch(lab) == "pre"]),
                          minlength=K)
    pre_use = pre_use / pre_use.sum()
    rare = np.flatnonzero(pre_use < min_usage)
    rows = []
    for lab in labs:
        z = Z[lab]
        zz = z[z >= 0]
        r = {"label": lab, "animal": lab.split("_")[0], "epoch": session_epoch(lab)}
        u = np.bincount(zz, minlength=K) / max(len(zz), 1)
        for k in live:
            r[f"use_{k}"] = u[k]
        runs = np.split(z, np.flatnonzero(np.diff(z) != 0) + 1)
        seq = [(int(q[0]), len(q)) for q in runs if q[0] >= 0]
        for k in live:
            dk = [n for s, n in seq if s == k]
            r[f"dwell_{k}"] = float(np.median(dk)) / am.FS_HZ if dk else np.nan
        r["dwell_median_s"] = float(np.median([n for _s, n in seq])) / am.FS_HZ if seq else np.nan
        C = np.zeros((K, K))
        for (a_, _), (b_, _) in zip(seq[:-1], seq[1:]):
            C[a_, b_] += 1
        P = C / np.maximum(C.sum(1, keepdims=True), 1)
        p = C.sum(1) / max(C.sum(), 1)
        r["entropy_bits"] = float(sum(p[i] * -(q[q > 0] * np.log2(q[q > 0])).sum() for i, q in enumerate(P)))
        r["rare_pre_share"] = float(np.isin(zz, rare).mean())
        rows.append(r)
    return pd.DataFrame(rows), names


def group_stats(tag: str, ref_tag: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`syllable_stats` COMBINED INTO THE REVIEWED GROUPS (Priya 11-17: merges are applied as groups): every state of
    ``tag`` joins the group it overlaps most (`syllable_stats` names; states below min usage are dropped), then per
    session: usage and median bout (s) per group (runs of the same group collapsed) and the group transition entropy.
    Returns (per-session table, state -> group map)."""
    from wfield_local import arhmm_model as am
    from wfield_local import arhmm_states as st
    _S, N = syllable_stats(tag, ref_tag)
    gn = list(state_groups(ref_tag))
    lut = np.full(64, -1)
    for r in N.itertuples():
        lut[int(r.state)] = gn.index(r.group) if r.group in gn else -1
    d = st.fits_dir() / f"states_{tag}"
    rows = []
    for p in sorted(d.glob("*.npz")):
        lab = p.stem
        if not session_epoch(lab):
            continue
        z = np.load(p)["z"].astype(int)
        g = np.where(z >= 0, lut[np.maximum(z, 0)], -1)
        gg = g[g >= 0]
        r = {"label": lab, "animal": lab.split("_")[0], "epoch": session_epoch(lab)}
        seq = [(int(q[0]), len(q)) for q in np.split(g, np.flatnonzero(np.diff(g) != 0) + 1) if q[0] >= 0]
        for k, n in enumerate(gn):
            r[f"use:{n}"] = float(np.mean(gg == k)) if len(gg) else np.nan
            dk = [m for s, m in seq if s == k]
            r[f"bout:{n}"] = float(np.median(dk)) / am.FS_HZ if dk else np.nan
        C = np.zeros((len(gn), len(gn)))
        for (a_, _), (b_, _) in zip(seq[:-1], seq[1:]):
            C[a_, b_] += 1
        P = C / np.maximum(C.sum(1, keepdims=True), 1)
        pw = C.sum(1) / max(C.sum(), 1)
        r["entropy_bits"] = float(sum(pw[i] * -(q[q > 0] * np.log2(q[q > 0])).sum() for i, q in enumerate(P)))
        rows.append(r)
    return pd.DataFrame(rows), N


def tim_trials(frame: str = "within") -> pd.DataFrame:
    """The per-trial TIM table (`task_independent_movement`, every session, ``frame`` = 'within' | 'frozen') joined
    with the trial's GLM-HMM role (argmax state; NaN for trials the GLM-HMM table lacks), restricted to the analysis
    sessions (epoch / August-only pre as everywhere here)."""
    from wfield_local import task_independent_movement as tim
    files = sorted((Path(tim.out_dir()) / "sessions").glob(f"*_{frame}.csv.gz"))
    D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    D["epoch"] = D.label.map(session_epoch)
    D = D[D.epoch.notna() & D.trial_id.notna()].copy()
    D["trial_id"] = D.trial_id.astype(int)
    T = glmhmm_table()
    roles = glmhmm_roles(T)
    pc = _pcols(T)
    T["role"] = [roles[int(pc[i][7:])] for i in T[pc].to_numpy().argmax(1)]
    D = D.merge(T[["label", "trial_id", "role"]], on=["label", "trial_id"], how="left")
    return D


def frozen_distance(prefix: str = "pre_s1_f") -> pd.DataFrame:
    """How LIKELY each session's movement is under the pre-stroke AR-HMM, with an OUT-OF-SAMPLE pre-stroke floor.

    Inputs: ``ll_<prefix><k>.csv`` beside the fold models in `arhmm_states.fits_dir()` (`arhmm_model score`); fold k
    left out each animal's k-th August pre day (model['held_out']). Each fold model has its own PCA basis, so a
    likelihood is compared only inside one model: for fold k and animal a, every session the model did NOT fit (the
    animal's held-out pre day and all its post days) is taken relative to their mean (the anchor A_ka). A post day's
    score = mean over folds of LL_k - A_ka; a pre day's score = LL_j - A_ja under the one fold j that held it out. The
    anchor cancels in the change vs pre, so `orofacial_cohort.epoch_tables` gives the change from each animal's own
    (held-out) pre days. Units: log-likelihood per frame (nats); negative change = less like the pre-stroke repertoire.
    """
    from wfield_local import arhmm_states as st
    L = pd.concat([pd.read_csv(f) for f in sorted(st.fits_dir().glob(f"ll_{prefix}*.csv"))])
    L = L[~L.in_fit.astype(bool)].copy()
    L["animal"] = L.label.str.split("_").str[0]
    L["epoch"] = L.label.map(session_epoch)
    L = L[L.epoch.notna()]
    L = L[(L.epoch != "pre") | L.held_out.astype(bool)]                 # pre days only where held out
    L["rel"] = L.ll_per_frame - L.groupby(["model", "animal"]).ll_per_frame.transform("mean")
    S = L.groupby(["label", "animal", "epoch"]).agg(ll_rel=("rel", "mean"), n_models=("model", "nunique")).reset_index()
    return S


def frozen_distance_frames(prefix: str = "pre_s1_f", tag: str | None = None) -> pd.DataFrame:
    """`frozen_distance` from the PER-FRAME scores (``llframe_<model>/<label>.npz``, `arhmm_model score --per-frame`),
    split by the reviewed syllable group of each frame (reference fit `tag`): one row per session with ``ll_rel`` (all
    frames, = `frozen_distance`), ``ll_rel_moving`` (every group except quiet: the STILLNESS CONTROL -- quiet frames
    are easy to predict, so a stiller session scores more pre-like) and ``ll_rel:<group>`` (how pre-like each kind of
    movement is). Same within-model anchor and held-out-pre rule as `frozen_distance`."""
    from wfield_local import arhmm_states as st
    tag = tag or st.REFERENCE_TAG
    groups = state_groups(tag)
    lut, gn = _group_lut(groups), list(groups)
    fits = st.fits_dir()
    L = pd.concat([pd.read_csv(f) for f in sorted(fits.glob(f"ll_{prefix}*.csv"))])
    L = L[~L.in_fit.astype(bool)].copy()
    L["animal"], L["epoch"] = L.label.str.split("_").str[0], L.label.map(session_epoch)
    L = L[L.epoch.notna() & ((L.epoch != "pre") | L.held_out.astype(bool))]
    ref = {}
    rows = []
    for r in L.itertuples():
        if r.label not in ref:
            z = np.load(fits / f"states_{tag}" / f"{r.label}.npz")["z"].astype(int)
            ref[r.label] = np.where(z >= 0, lut[np.maximum(z, 0)], -1)
        g = ref[r.label]
        lp = np.load(fits / f"llframe_{r.model}" / f"{r.label}.npz")["lp"].astype(float)
        ok = np.isfinite(lp) & (g >= 0)
        row = {"model": r.model, "label": r.label, "animal": r.animal, "epoch": r.epoch, "all": lp[ok].mean(),
               "moving": lp[ok & (g != gn.index("quiet"))].mean() if "quiet" in gn else np.nan}
        for k, n in enumerate(gn):
            m = ok & (g == k)
            row[n] = lp[m].mean() if m.sum() >= 50 else np.nan
        rows.append(row)
    F = pd.DataFrame(rows)
    cols = ["all", "moving", *gn]
    for c in cols:
        F[c] = F[c] - F.groupby(["model", "animal"])[c].transform("mean")
    S = F.groupby(["label", "animal", "epoch"])[cols].mean().reset_index()
    return S.rename(columns={"all": "ll_rel", "moving": "ll_rel_moving", **{n: f"ll_rel:{n}" for n in gn}})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("cmd", choices=("collect",))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args(argv)
    collect(a.tag, a.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

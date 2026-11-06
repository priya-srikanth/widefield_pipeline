"""The map-cache key hashes only the defaults.yaml keys the map builders read (rebuild 2026-11-03, task 3).

`map_collect.cache_kind` used to hash the whole of defaults.yaml, so an unrelated edit (a deck option, a DLC parameter,
the rebuild's joint_basis / trial_selection blocks) cold-started every per-session map cache. Now it hashes
`map_collect.MAP_YAML_KEYS`. These tests pin both directions, and re-read the builder modules so a builder that starts
reading another defaults.yaml section fails here instead of serving a stale map.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from wfield_local import config, map_collect

REPO = Path(__file__).resolve().parents[1]

#: modules the per-session map builders (beta_maps.session_maps, position_reference_maps.session_raw_maps) call into
BUILDER_MODULES = ("beta_maps", "position_reference_maps", "locanmf_position_decoder", "locanmf_frozen_decoder",
                   "joint_basis", "quiet_periods", "rest_engagement", "rest_by_position", "precue_engagement_states",
                   "lick_detection", "behavior_position", "block_ids", "plot_spout_trial_averages", "daq_io",
                   "framemap_event_maps", "lp_licks", "gate_source", "behavior_events", "analysis_kit",
                   "locanmf_cue_lick_analysis", "plot_lick_aligned_averages", "locanmf_lick_aligned", "nolick_decoder")
#: sections read by those modules whose effect reaches the cache KIND by another route (run_mode / effective_kind)
VIA_KIND = {"trial_selection"}
#: sections a builder MODULE reads only in code no map builder reaches (classified by hand, as the test asks):
#: precue_engagement_states.late_fraction (Priya 11-06, PS93 camera late window) feeds the state discriminator's
#: training rows in run_animal; the map builders import only `engagement_gate` from that module, which never reads it
OUTSIDE_MAPS = {"precue_engagement_states"}
#: config.py helpers the builders call -> the key path each reads
CONFIG_HELPER_KEYS = {"preprocess.maps.tag", "hemo.variant"}


def _key(defaults, monkeypatch):
    monkeypatch.setattr(config, "defaults", lambda session=None: defaults)
    return map_collect.cache_kind("prm", "cue", "working", {"post_s": 2.0}, "code")


@pytest.mark.parametrize("path, value", [
    (("deck",), {"anything": 1}),
    (("joint_basis", "pins", "PS92"), {"id": "000000000000", "hemo_variant": "x"}),
    (("trial_selection", "window"), "trial"),
    (("hemo", "build_variant"), "something_else"),
    (("preprocess", "svd", "k"), 7),
    (("preprocess", "maps", "cue_pre_s"), 9.0),
    (("locanmf", "maxrank"), 99),
    (("dlc",), None),
    (("new_unrelated_block",), {"a": 1}),
])
def test_unrelated_keys_do_not_move_the_key(monkeypatch, path, value):
    base = copy.deepcopy(config.defaults())
    k0 = _key(base, monkeypatch)
    d = copy.deepcopy(base)
    node = d
    for p in path[:-1]:
        node = node.setdefault(p, {})
    node[path[-1]] = value
    assert _key(d, monkeypatch) == k0, path


@pytest.mark.parametrize("path, value", [
    (("decode", "max_rt_s"), 1.234),
    (("decode", "precue_lickfree"), False),
    (("lick_detection", "min_ili_ms"), 41),
    (("segmentation", "rest", "variant"), "zz"),
    (("preprocess", "maps", "tag"), "affine9"),
    (("hemo", "variant"), "meegkit_hpfit_lag_refitT_swapmask"),
])
def test_builder_keys_move_the_key(monkeypatch, path, value):
    base = copy.deepcopy(config.defaults())
    k0 = _key(base, monkeypatch)
    d = copy.deepcopy(base)
    node = d
    for p in path[:-1]:
        node = node.setdefault(p, {})
    node[path[-1]] = value
    assert _key(d, monkeypatch) != k0, path


def test_builder_modules_read_only_covered_sections():
    covered = {p[0] for p in map_collect.MAP_YAML_KEYS if len(p) == 1}
    #: sections covered only by a SUB-path: an access must spell exactly that chain
    partial = {p[0]: "".join(f'["{k}"]' for k in p) for p in map_collect.MAP_YAML_KEYS if len(p) > 1}
    pat = re.compile(r"defaults\(\)\s*(?:\[|\.get\()\s*[\"']([a-z_0-9]+)[\"']")
    read = {}
    for m in BUILDER_MODULES:
        src = (REPO / "wfield_local" / f"{m}.py").read_text(encoding="utf-8")
        for ln in src.splitlines():
            for sec in pat.findall(ln.split("#", 1)[0]):
                if sec in partial and partial[sec] in ln.replace("'", '"'):
                    continue
                read.setdefault(sec, set()).add(m)
        # every other form of access (a bare `config.defaults()` bound to a name) would escape the regex: refuse it
        for ln in src.splitlines():
            code = ln.split("#", 1)[0]
            if "defaults()" in code and not pat.search(code) and "def defaults" not in code:
                pytest.fail(f"{m}: unrecognised defaults() access, classify it by hand: {ln.strip()}")
    unexpected = {s: ms for s, ms in read.items() if s not in covered | VIA_KIND | OUTSIDE_MAPS}
    assert not unexpected, unexpected


def test_config_helpers_the_builders_use_are_covered():
    have = {".".join(p) for p in map_collect.MAP_YAML_KEYS}
    assert CONFIG_HELPER_KEYS <= have


def test_prm_digest_covers_cross_module_helpers():
    """Review M4 (11-03, second pass): read WHAT `_raw_maps_code` HASHES (`_raw_maps_parts`) and check that every
    cross-module helper its hashed sources call (`quiet_periods.<f>`, `ak.<f>` / `analysis_kit.<f>`, `daq_io.<f>`,
    `be.<f>` / `behavior_events.<f>`, and from-imports of them) is itself hashed: closure() follows same-module names
    only, so a cross-module helper must be listed as a root."""
    import inspect

    from wfield_local import position_reference_maps as prm
    parts = prm._raw_maps_parts()
    srcs = [p.source() if hasattr(p, "source") else inspect.getsource(p) for p in parts]
    blob = "\n".join(srcs)
    alias = {"quiet_periods": "quiet_periods", "ak": "analysis_kit", "analysis_kit": "analysis_kit",
             "daq_io": "daq_io", "be": "behavior_events", "behavior_events": "behavior_events"}
    want = set()
    for s in srcs:
        for a, f in re.findall(r"\b(quiet_periods|ak|analysis_kit|daq_io|be|behavior_events)\.(\w+)\(", s):
            want.add((alias[a], f))
        for mod, names in re.findall(r"from wfield_local\.(quiet_periods|analysis_kit|daq_io|behavior_events) "
                                     r"import ([\w, ]+)", s):
            want |= {(mod, n.strip()) for n in names.split(",") if n.strip()}
    missing = sorted(f"{m}.{f}" for m, f in want if not re.search(rf"^def {f}\(", blob, re.M))
    assert not missing, missing

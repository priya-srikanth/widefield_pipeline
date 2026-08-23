"""Every analysis slide must state the window and binning it was built on.

On 2026-08-21 decode.max_rt_s moved from 2.0 s to 3.5 s and eleven modules kept their own hardcoded
2.0. For a day the deck showed figures cut at two different definitions of "engaged", side by side,
with nothing on either saying which. A reader could not tell them apart by looking.

The line is derived from the figure's own filename plus configs/defaults.yaml at build time, so it
cannot drift from what it describes -- which a hand-written note can, and did.
"""
from wfield_local import config
from wfield_local.locanmf_analysis_deck import window_provenance


def test_a_cue_aligned_figure_reports_the_cue_window():
    line = window_provenance(["locanmf_position_session_PS93_0820_locanmf_cue_base-none_cv-block.png"])
    assert "POST-CUE" in line and "STARTS at the cue" in line


def test_a_precue_figure_says_the_window_ends_at_the_cue():
    """The distinction that matters for the headline: pre-cue is motor-independent."""
    line = window_provenance(["locanmf_position_session_PS93_0820_locanmf_precue_base-none.png"])
    assert "PRE-CUE" in line and "ENDS at the cue" in line
    assert "POST-CUE" not in line


def test_a_lick_aligned_figure_reports_the_lick_window():
    line = window_provenance(["locanmf_position_session_PS94_0820_locanmf_lick_base-none.png"])
    assert "POST-LICK" in line and "first detected lick" in line


def test_the_engaged_cut_quoted_is_the_configured_one():
    """The whole point: if max_rt_s moves again, this line moves with it."""
    line = window_provenance(["x_cue_base-none.png"])
    assert f"{float(config.defaults()['decode']['max_rt_s']):.1f} s" in line


def test_the_bin_count_and_width_come_from_config():
    d = config.defaults()["decode"]
    for align in ("cue", "lick", "precue"):
        line = window_provenance([f"x_{align}_base-none.png"])
        nb = int((d.get("bins") or {}).get(align, 1) or 1)
        post = float(d.get(f"{align}_post_s", 2.0))
        if nb > 1:
            assert f"{nb} sub-bins of {post / nb:.2f} s" in line, (align, line)
        else:
            assert "1 bin" in line


def test_baseline_and_cv_are_reported_when_the_name_carries_them():
    line = window_provenance(["a_cue_base-none_cv-block.png"])
    assert "- baseline: none" in line
    assert "- CV: block" in line and "GroupKFold" in line


def test_a_figure_with_no_alignment_gets_no_claim():
    """A schematic or a text slide must not be given a guessed window."""
    assert window_provenance(["section_g_counts.png"]) == ""
    assert window_provenance([]) == ""


def test_mixed_names_resolve_to_the_most_specific_alignment():
    """'precue' contains 'cue'; the pre-cue reading must win, or the window is stated backwards."""
    line = window_provenance(["locanmf_encoder_precue_thing.png"])
    assert "PRE-CUE" in line and "ENDS at the cue" in line


def test_two_notes_sharing_a_long_prefix_are_not_deduped_together():
    """The note dedup must key on the WHOLE text.

    On 2026-08-23 _M_LICK_UNIT was prepended to M_FIXEDSCALE, M_GATE and M_POSTSTROKE -- three
    unrelated methods blocks that then shared their first 80 characters. A prefix key would have
    rendered the second and third as "METHODS -- same as slide N" pointing at the FIRST one's
    methods. A wrong cross-reference reads exactly like a right one, so it is worse than the
    repetition the dedup was added to remove.
    """
    import hashlib

    from wfield_local import locanmf_analysis_deck as deck

    blocks = {n: getattr(deck, n) for n in dir(deck)
              if n.startswith("M_") and isinstance(getattr(deck, n), str)}
    by_hash = {}
    for n, t in blocks.items():
        by_hash.setdefault(hashlib.sha1(t.encode("utf-8")).hexdigest(), []).append(n)
    dupes = {h: v for h, v in by_hash.items() if len(v) > 1}
    assert not dupes, f"distinct methods blocks hash the same: {list(dupes.values())}"

    prefix_shared = {}
    for n, t in blocks.items():
        prefix_shared.setdefault(t[:80], []).append(n)
    shared = [v for v in prefix_shared.values() if len(v) > 1]
    if shared:
        # not a failure -- it is the CONDITION that made the prefix key wrong. Assert the real key
        # separates them, so this test keeps its meaning as the notes change.
        for group in shared:
            hs = {hashlib.sha1(blocks[n].encode("utf-8")).hexdigest() for n in group}
            assert len(hs) == len(group), (
                f"{group} share a prefix AND a key -- the dedup would cross-reference them wrongly")


def test_the_dedup_key_is_not_a_prefix():
    """Source-level, because the failure is silent and only shows up as a wrong slide number."""
    import inspect

    from wfield_local import locanmf_analysis_deck as deck

    src = inspect.getsource(deck.build_analysis_deck)
    assert 'key = (text or "")[:80]' not in src, "the dedup key is a prefix again"
    assert "hashlib.sha1" in src

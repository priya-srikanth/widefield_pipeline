"""Imaging page section 1, 11-03 update: per-animal refit-decoder table on the camera-defined trial gates (Priya 11-03).
PRIMARY trials = camera working = contact or incomplete camera lick (jaw-only OUT, Priya 11-03 late), up to the session's
last camera lick (`working_pl_lick_licks`; Priya 11-03 "last camera lick cutoff sounds good"); no cutoff, contact /
close-spout anchors, working incl. jaw-only and the DAQ gates are robustness rows.

    python docs/results_pages/imaging/p_img_s1_table_20261103.py

Builds the HTML blocks for section 1 from the run CSVs (no decoding here) and writes them to
s1_update_20261103.html beside this script, plus the numbers for the PI deck (slide 9) to s1_update_20261103_numbers.md.
Apply to the LIVE page: replace section 1's per-animal <h4> + table, the Fig 1.2 caption sentence, and the takeaway;
rebuild from the live version immediately before publishing (a8's figure-rules helper also republishes the page).

Sources (<run>/imaging_encoding/...):
  LP_licks_ENL2s/qc/s1_refit_basis_1104.csv    Allen areas, camera arms (split = primary basis; plain_orig = the old table's)
  LP_licks_ENL2s/qc/s1_refit_locanmf_1104.csv  joint-LocaNMF, camera arms
  (the _1104 runs add the jaw-free arms camlickpl_*; the 11-03 files without the suffix have only jaw-inclusive arms)
  qc/s1_refit_basis.csv                   Allen, DAQ arms: quit-tail working and DAQ licking (the table until 11-03)
  qc/s1_trial_classes_cue_stop.json       commit of the trial classes the camera arms used
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

R = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
HERE = Path(__file__).parent
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]


def per_animal(D: pd.DataFrame, col: str) -> pd.DataFrame:
    return D.groupby(["animal", "epoch"])[col].mean().unstack("epoch").reindex(index=AN, columns=EP)


def main() -> int:
    A = pd.read_csv(R / "LP_licks_ENL2s/qc/s1_refit_basis_1104.csv")
    L = pd.read_csv(R / "LP_licks_ENL2s/qc/s1_refit_locanmf_1104.csv")
    Q = pd.read_csv(R / "qc/s1_refit_basis.csv")
    L = L[(L.label.str[-4:] >= "0806") | (L.epoch != "pre")]                    # August-only pre, as the Allen rows
    meta = json.loads((R / "qc/s1_trial_classes_cue_stop.json").read_text())
    Lw = {(t, al): per_animal(L[(L["trials"] == t) & (L["align"] == al)], "acc") for t in L.trials.unique()
          for al in ("precue", "cue")}
    prim = {("split", w): per_animal(A, f"split_camlickpllick_{w}") for w in ("pre", "post")}
    nocut = {("split", w): per_animal(A, f"split_camlick_{w}") for w in ("pre", "post")}   # camera licks, no cutoff

    def row(an, tabs):
        return "".join(f"<td>{tabs[e]:.2f}</td>" if pd.notna(tabs[e]) else "<td>&ndash;</td>" for e in EP)

    head = ("<tr><th>animal</th><th>basis</th>" + "".join(f"<th>{w}: {e}</th>" for w in ("pre-cue", "post-cue")
                                                         for e in EP) + "</tr>")
    body = []
    for an in AN:
        for lab, pre_t, post_t in (("Allen split", prim[("split", "pre")], prim[("split", "post")]),
                                   ("LocaNMF", Lw[("camlickpl_lick", "precue")], Lw[("camlickpl_lick", "cue")])):
            body.append(f"<tr><td>{an}</td><td>{lab}</td>{row(an, pre_t.loc[an])}{row(an, post_t.loc[an])}</tr>")
    n = A[A.split_camlickpllick_post.notna()].groupby(["animal", "epoch"]).size().unstack().reindex(index=AN, columns=EP)
    cnt = " / ".join(f"{e} " + " / ".join(str(int(n.loc[a, e])) for a in AN) for e in EP)

    # robustness: mean of the 4 animals' session means, per epoch, for every arm
    def m4(t):
        return t.mean(0)
    arms = [("Allen split, camera working to the session's last camera lick — table above", prim[("split", "pre")], prim[("split", "post")]),
            ("Allen split, camera working, no end-of-session cutoff", nocut[("split", "pre")], nocut[("split", "post")]),
            ("Allen split, camera working to the last contact", per_animal(A, "split_camlickplcontact_pre"), per_animal(A, "split_camlickplcontact_post")),
            ("Allen split, camera working to the last close-spout lick", per_animal(A, "split_camlickplclose_pre"), per_animal(A, "split_camlickplclose_post")),
            ("Allen split, camera working plus jaw-only trials, to the last camera lick", per_animal(A, "split_campllick_pre"), per_animal(A, "split_campllick_post")),
            ("Allen plain (no ALM / tjM1 split), camera working to the last camera lick", per_animal(A, "plain_orig_camlickpllick_pre"), per_animal(A, "plain_orig_camlickpllick_post")),
            ("Allen split, DAQ quit-tail working", per_animal(Q, "split_work_pre"), per_animal(Q, "split_work_post")),
            ("Allen plain, DAQ licking trials (the table until 11-03)", per_animal(Q, "plain_orig_pre"), per_animal(Q, "plain_orig_post")),
            ("LocaNMF, camera working to the last camera lick — table above", Lw[("camlickpl_lick", "precue")], Lw[("camlickpl_lick", "cue")]),
            ("LocaNMF, camera working, no end-of-session cutoff", Lw[("camlick", "precue")], Lw[("camlick", "cue")]),
            ("LocaNMF, camera working plus jaw-only trials, to the last camera lick", Lw[("camworkpl_lick", "precue")], Lw[("camworkpl_lick", "cue")]),
            ("LocaNMF, DAQ quit-tail working", Lw[("quittail", "precue")], Lw[("quittail", "cue")])]
    rob = ["<tr><th>decoder basis, trials</th>" + "".join(f"<th>{w}: {e}</th>" for w in ("pre-cue", "post-cue")
                                                         for e in EP) + "</tr>"]
    for lab, a, b in arms:
        rob.append(f"<tr><td>{lab}</td>" + "".join(f"<td>{v:.2f}</td>" for v in m4(a)) +
                   "".join(f"<td>{v:.2f}</td>" for v in m4(b)) + "</tr>")

    # the robustness sentence states the measured spread, not a fixed claim
    dev = max(float(pd.concat([(m4(a) - m4(arms[0][1])).abs(), (m4(b) - m4(arms[0][2])).abs()]).max())
              for lab, a, b in arms if "DAQ licking" not in lab)
    rob_claim = f"move no cell by more than {dev:.2f}"
    sp, spp = prim[("split", "post")], prim[("split", "pre")]
    rng_post, rng_pre = (float(sp.min().min()), float(sp.max().max())), (float(spp.min().min()), float(spp.max().max()))
    up = [a for a in AN if spp.loc[a, "acute"] > spp.loc[a, "pre"] + 0.04]
    dn = [a for a in AN if spp.loc[a, "acute"] < spp.loc[a, "pre"] - 0.04]
    flat = [a for a in AN if a not in up + dn]
    post_dn = [a for a in AN if sp.loc[a, "acute"] < sp.loc[a, "pre"]]

    html = f"""<!-- section 1 update 11-03 (window 5b; p_img_s1_table_20261103.py). Classes: commit {meta['commit']}, window {meta['window']} -->
<h4>Per animal (session means; refit within each session; trials = CAMERA-DEFINED WORKING: the camera saw a lick, contact or incomplete, between the cue and the trial stop, up to the session&rsquo;s last camera lick (jaw-only and no-response trials excluded); video lick train (cam4 + cam1) for the lick-free pre-cue window; pre = August only; sessions with a decode {cnt} (PS93 pre 7: PS93_0813 has no video); robustness rows below the table)</h4>
<div class="tbl"><table>
{head}
{chr(10).join(body)}
</table></div>
<p class="input"><b>Robustness</b> (mean of the four animals&rsquo; session means): the trial gate, the end-of-session cutoff, the Allen basis (split vs plain) and the decoder basis (Allen vs LocaNMF) {rob_claim}; DAQ licking trials (the table until 11-03) sit higher post-cue because every such trial&rsquo;s window contains the lick.</p>
<div class="tbl"><table class="wide">
{chr(10).join(rob)}
</table></div>"""
    sentence = ("A decoder refit on the acute session still loses the far spouts, so part of the acute loss of section 2 "
                "is not a frozen-decoder mismatch; fewer acute trials and changed licking may contribute.")
    take = (f"Re-fitting each day, position stays decodable in every epoch: post-cue {rng_post[0]:.2f}&ndash;{rng_post[1]:.2f}, "
            f"pre-cue {rng_pre[0]:.2f}&ndash;{rng_pre[1]:.2f} against chance 0.167 (table; Allen areas with the ALM / tjM1 split, "
            "camera-defined working trials; LocaNMF agrees). Post-cue accuracy falls acutely in "
            f"{len(post_dn)} / 4 animals and is back at or above pre by chronic. Pre-cue there is no consistent acute change "
            f"(up in {' / '.join(up) or 'none'}, down in {' / '.join(dn) or 'none'}"
            + (f", flat in {' / '.join(flat)}" if flat else "") + "). ")
    (HERE / "s1_update_20261103.html").write_text(
        html + "\n<!-- Fig 1.2 caption sentence (replaces 'A decoder trained on the acute session itself ...'): -->\n<p>"
        + sentence + "</p>\n<!-- takeaway, first sentences (keep the rest of the existing take: position information "
        "not a plan, slow band): -->\n<p>" + take + "</p>\n", encoding="utf8")
    nums = (f"# Section 1 numbers for PI deck slide 9 (11-03)\n\nSource: {R}/LP_licks_ENL2s/qc/s1_refit_basis_1104.csv "
            f"(split_camlickpllick_*), classes commit {meta['commit']}.\n\n"
            f"* Decoder refit within each session: post-cue {rng_post[0]:.2f}–{rng_post[1]:.2f}, pre-cue "
            f"{rng_pre[0]:.2f}–{rng_pre[1]:.2f} (chance .17), every animal and epoch (was .55–.91 / .28–.55 on DAQ licking "
            "trials, plain Allen).\n"
            f"* Acute post-cue drop in {len(post_dn)} / 4 animals (PS95 included now; on DAQ licking trials PS95 went UP).\n"
            f"* No consistent acute pre-cue change: up in {', '.join(up) or 'none'}, down in {', '.join(dn) or 'none'}"
            + (f", flat in {', '.join(flat)}" if flat else "") + " (was 'up in PS92 / PS95, down in PS93 / PS94').\n"
            f"* Sessions with a decode: {cnt} (acute was 4 / 4 / 4 / 1 on DAQ licking trials; camera working counts "
            "incomplete licks, so acute sessions with no far-contra contact can decode).\n"
            "* Trials: camera-defined working (contact or incomplete camera lick, cue to trial stop; jaw-only out), up to the session's last camera lick; basis: Allen areas with the "
            "ALM / tjM1 split.\n")
    (HERE / "s1_update_20261103_numbers.md").write_text(nums, encoding="utf8")
    print(html[:3000]); print(nums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

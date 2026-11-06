from pathlib import Path

p = Path(r"C:\Users\SabatiniLab\AppData\Local\Temp\claude\C--Users-SabatiniLab\c67bec01-3925-49ff-9cdc-8ff295df4bfa\scratchpad\imaging_page\imaging_results.html")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep('<span class="st prov">superseded &mdash; being re-rendered as within-animal differences (pre = 0) instead of ratios; numbers in 9f&ndash;9k will change</span> ', '')
rep("Each animal divided by its own held-out pre (pre = 1), pooled over animals; text = change vs pre",
    "Each animal minus its own held-out pre (pole units, pre = 0; 1 = the full not-P &rarr; P separation), pooled over animals; text = change vs pre")
rep(" Far contra: acute &minus;1.12 [&minus;2.46, +.04] 3 / 4 (PS95 one session), subacute &minus;.39 [&minus;.65, &minus;.12]* 4 / 4, chronic &minus;.25 [&minus;.52, +.01] 4 / 4. Close centre is dominated by PS93 acute (&minus;89): its held-out pre is near 0, so the ratio blows up &mdash; not a real effect.",
    " Far contra: acute &minus;.81 [&minus;1.69, +.02] 3 / 4 (PS95 one session, &minus;2.06), subacute &minus;.28 [&minus;.45, &minus;.09]* 4 / 4, chronic &minus;.18 [&minus;.37, +.01] 4 / 4. Far middle chronic &minus;.51* 4 / 4. (Re-rendered 10-25 as differences; the earlier ratio version was distorted by a near-zero pre in close centre.)")
rep("Far contra: acute &minus;.27 [&minus;.64, +.08] 3 / 4, subacute &minus;.18 [&minus;.34, &minus;.03]* 4 / 4, chronic &minus;.15 [&minus;.39, +.15] 3 / 4.",
    "Far contra: acute &minus;.21 [&minus;.49, +.07] 3 / 4, subacute &minus;.15 [&minus;.26, &minus;.03]* 4 / 4, chronic &minus;.13 [&minus;.32, +.11] 3 / 4.")
rep("Each animal&rsquo;s session-mean trace / its pre window mean, then mean",
    "Each animal&rsquo;s session-mean trace minus its pre window mean, then mean")
rep("No consistent change (far contra acute +.44 [&minus;.28, +1.56], 2 / 4 below pre; chronic &minus;.15, 3 / 4).",
    "No consistent change (far contra acute +.39 [&minus;.31, +1.40], 2 / 4 below pre, driven by PS94; chronic &minus;.15, 3 / 4).")
p.write_text(s, encoding="utf-8")
print("ok")

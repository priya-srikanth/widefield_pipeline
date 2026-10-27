from pathlib import Path

p = Path(r"C:\Users\SabatiniLab\AppData\Local\Temp\claude\C--Users-SabatiniLab\c67bec01-3925-49ff-9cdc-8ff295df4bfa\scratchpad\imaging_page\imaging_results.html")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


NOTE = ("Each animal divided by its own held-out pre (pre = 1), pooled over animals; text = change vs pre "
        "(paired animals &rarr; sessions bootstrap, * = 95 % CI excludes 0) and animals below pre / 4. "
        "Pre = August only (~10 days), so like Fig 13c this family cannot show 2-month drift.")
anchor = '<figcaption><b class="h">Fig 9e &middot; leave-one-out mean trajectories, cue-aligned.</b></figcaption></figure>'
rep(anchor, anchor + "\n" + "\n".join([
    '<figure id="fig9f"><img src="cp_window_precue.png" alt="CD leave-one-out window means, pre-cue, pooled within animal" loading="lazy"><figcaption><b class="h">Fig 9f &middot; pooled within animal, pre-cue window (pooled twin of Fig 9c).</b> '
    + NOTE + ' Far contra: acute &minus;1.12 [&minus;2.46, +.04] 3 / 4 (PS95 one session), subacute &minus;.39 [&minus;.65, &minus;.12]* 4 / 4, chronic &minus;.25 [&minus;.52, +.01] 4 / 4. Close centre is dominated by PS93 acute (&minus;89): its held-out pre is near 0, so the ratio blows up &mdash; not a real effect.</figcaption></figure>',
    '<figure id="fig9g"><img src="cp_window_cue.png" alt="CD leave-one-out window means, cue, pooled within animal" loading="lazy"><figcaption><b class="h">Fig 9g &middot; pooled within animal, cue window (twin of Fig 9d).</b> Far contra: acute &minus;.27 [&minus;.64, +.08] 3 / 4, subacute &minus;.18 [&minus;.34, &minus;.03]* 4 / 4, chronic &minus;.15 [&minus;.39, +.15] 3 / 4.</figcaption></figure>',
    '<figure id="fig9h"><img src="cp_traj_cue.png" alt="CD leave-one-out trajectories, cue, pooled within animal" loading="lazy"><figcaption><b class="h">Fig 9h &middot; pooled within animal, cue-aligned trajectories (twin of Fig 9e).</b> Each animal&rsquo;s session-mean trace / its pre window mean, then mean &plusmn; SEM over animals; shaded = window used for the means.</figcaption></figure>',
    '<figure id="fig9i"><img src="cp_traj_precue.png" alt="CD leave-one-out trajectories, pre-cue, pooled" loading="lazy"><figcaption><b class="h">Fig 9i &middot; pooled within animal, pre-cue-aligned trajectories.</b></figcaption></figure>',
    '<figure id="fig9j"><img src="cp_window_lick.png" alt="CD leave-one-out window means, lick, pooled" loading="lazy"><figcaption><b class="h">Fig 9j &middot; pooled within animal, lick window.</b> No consistent change (far contra acute +.44 [&minus;.28, +1.56], 2 / 4 below pre; chronic &minus;.15, 3 / 4).</figcaption></figure>',
    '<figure id="fig9k"><img src="cp_traj_lick.png" alt="CD leave-one-out trajectories, lick, pooled" loading="lazy"><figcaption><b class="h">Fig 9k &middot; pooled within animal, lick-aligned trajectories.</b></figcaption></figure>',
]))

rep('Panels: frozen, refit, gap; F_deficit, G_reorg, refit deficit. Triangles',
    'Panels: frozen, refit, gap; F_deficit, G_reorg, refit deficit. '
    '<b>F_deficit</b> = pre_frozen &minus; frozen: how much the old (pre-stroke) decoder stops working. '
    '<b>G_reorg</b> = (refit &minus; frozen) &minus; (pre_refit &minus; pre_frozen): information a newly fitted decoder finds that the old one cannot &mdash; moved, not lost. '
    '<b>Refit deficit</b> = pre_refit &minus; refit = F &minus; G: information missing even for a decoder allowed to relearn &mdash; genuinely weaker. '
    'Chronically refit is back (1.10 / 1.11 / 1.02 / .99 of pre), so the refit deficit is ~0. Triangles')
p.write_text(s, encoding="utf-8")
print("ok")

from pathlib import Path

p = Path(r"C:\Users\SabatiniLab\AppData\Local\Temp\claude\C--Users-SabatiniLab\c67bec01-3925-49ff-9cdc-8ff295df4bfa\scratchpad\imaging_page\imaging_results.html")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


POOL = ("Pooled over 4 animals; difference columns are within-animal (each animal minus its own pre), "
        "contours = nested bootstrap with max-statistic correction.")
anchor = '<figcaption><b class="h">Fig 7d &middot; pooled, lick window, LP lick gate.</b> Far contra 0.67 acute, back by subacute.</figcaption></figure>'
rep(anchor, anchor + "\n" + "\n".join([
    '<p class="input">Figs 7e&ndash;7j: the same position maps under the other references and gates (15r family). REST ref = each session&rsquo;s time-local rest baseline; PRECUE ref = each trial&rsquo;s own pre-cue window, so it shows the cue-evoked <i>increment</i> and removes the anticipatory position signal. PRECUE has no ENL panel by construction. A June-vs-August pre map-drift version is in progress (e3).</p>',
    '<figure id="fig7e"><img src="p_ct_pre_cue.png" alt="Position maps, pre-cue reference, cue window, contact gate" loading="lazy"><figcaption><b class="h">Fig 7e &middot; PRECUE reference, cue window, contact gate.</b> '
    + POOL + ' Acute amplitude / own pre: far contra 0.18, far middle 0.52, far ipsi 1.00, near positions 1.55&ndash;1.69. The cue-evoked increment falls more than the rest-referenced map (Fig 7b, 0.47); chronic far contra is above pre.</figcaption></figure>',
    '<figure id="fig7f"><img src="p_ct_rest_lick.png" alt="Position maps, rest reference, lick window, contact gate" loading="lazy"><figcaption><b class="h">Fig 7f &middot; REST reference, lick window, contact gate</b> (compare the LP-gated Fig 7d). ' + POOL + '</figcaption></figure>',
    '<figure id="fig7g"><img src="p_ct_pre_lick.png" alt="Position maps, pre-cue reference, lick window, contact gate" loading="lazy"><figcaption><b class="h">Fig 7g &middot; PRECUE reference, lick window, contact gate.</b> ' + POOL + '</figcaption></figure>',
    '<figure id="fig7h"><img src="p_lp_pre_lick.png" alt="Position maps, pre-cue reference, lick window, LP gate" loading="lazy"><figcaption><b class="h">Fig 7h &middot; PRECUE reference, lick window, LP lick gate.</b> ' + POOL + '</figcaption></figure>',
    '<figure id="fig7i"><img src="p_ct_rest_precue.png" alt="Position maps, rest reference, pre-cue window, contact gate" loading="lazy"><figcaption><b class="h">Fig 7i &middot; REST reference, pre-cue window, contact gate</b> (anticipatory position signal). ' + POOL + '</figcaption></figure>',
    '<figure id="fig7j"><img src="p_enl_rest.png" alt="Position maps, rest reference, ENL pre-cue window" loading="lazy"><figcaption><b class="h">Fig 7j &middot; REST reference, ENL (1 s lick-free pre-cue window), DAQ contact licks.</b> ' + POOL + ' Acute amplitude / own pre: far contra 1.67, far ipsi 1.10, far middle 1.00, near ipsi 1.72, near middle 0.42, near contra 0.55 &mdash; the far-contra loss seen after the cue is not present in the lick-free pre-cue window; chronic far contra falls below pre.</figcaption></figure>',
]))
p.write_text(s, encoding="utf-8")
print("ok")

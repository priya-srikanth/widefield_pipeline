from pathlib import Path

p = Path(r"C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\c67bec01-3925-49ff-9cdc-8ff295df4bfa\scratchpad\imaging_page\imaging_results.html")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


# ---- status table
rep('''<tr><td>Position falls ~5&times; more than behavioural state or running on the same cortex</td><td><span class="st stands">stands</span> (ratio)</td><td>engagement, DAQ running</td><td>pooled</td></tr>''',
    '''<tr><td>Position falls ~5&times; more than behavioural state or running on the same cortex; the pooled state drop is PS94 alone</td><td><span class="st stands">stands</span> (ratio) <span class="st corr">corrected 10-25</span></td><td>engagement, DAQ running</td><td>state intact in 3 / 4</td></tr>''')
rep('''<tr><td>Pre-stroke template amplitude drops and returns; its shape does not</td><td><span class="st stands">stands</span></td><td>contact</td><td>pooled only</td></tr>''',
    '''<tr><td>Pre-stroke template: amplitude drops in 4 / 4 but returns in only 2 / 4; its shape stays off in 4 / 4</td><td><span class="st corr">corrected 10-25</span></td><td>contact</td><td>4 / 4 drop; 2 / 4 return</td></tr>''')
rep('''<tr><td>Rest carries position; frozen rest decoder retained chronically</td><td><span class="st stands">stands</span></td><td>engagement</td><td>4 / 4 (rest &gt; null)</td></tr>''',
    '''<tr><td>Rest carries position; frozen rest decoder retained chronically</td><td><span class="st stands">stands</span></td><td>engagement</td><td>4 / 4 (rest &gt; null)</td></tr>
<tr><td>Mixed models: frozen and refit decoders recover with time and track the behavioural deficit; the reorganisation index keeps rising (drift-sized)</td><td><span class="st stands">stands</span> (rebuilt attempts table, 10-25)</td><td>contact or LP-attempt deficit</td><td>animal random effects</td></tr>''')

# ---- section 3: per-animal state table + correction + figure
rep('''State does fall (acute &minus;0.069 [&minus;0.148, &minus;0.012], subacute &minus;0.104, chronic &minus;0.092, Bonferroni intervals exclude 0), so the claim is the <b>ratio</b> &mdash; position falls about five times further &mdash; never that state is flat. No per-animal state table exists yet. Values from the 10-15 re-render; <code>BEHAVIOURAL_STATE_CONTROL.md</code> still quotes older numbers.</div>''',
    '''State does fall at the cohort level (acute &minus;0.069 [&minus;0.148, &minus;0.012], subacute &minus;0.104, chronic &minus;0.092, Bonferroni intervals exclude 0), so the claim is the <b>ratio</b> &mdash; position falls about five times further &mdash; never that state is flat. Values from the 10-15 re-render; <code>BEHAVIOURAL_STATE_CONTROL.md</code> still quotes older numbers.</div>
<h4>Per animal (added 10-25): state decoder above chance, (accuracy &minus; 1/3) / (2/3)</h4>
<div class="tbl"><table>
<tr><th>animal</th><th>pre</th><th>acute</th><th>subacute</th><th>chronic</th></tr>
<tr><td>PS92</td><td>.963</td><td>.939</td><td>.883</td><td>.873</td></tr>
<tr><td>PS93</td><td>.978</td><td>.949</td><td>.961</td><td>.951</td></tr>
<tr><td>PS94</td><td>.978</td><td>.747</td><td>.577</td><td>.447</td></tr>
<tr><td>PS95</td><td>.985</td><td>.979 (1 session)</td><td>.915</td><td>.960</td></tr>
</table></div>
<div class="take"><b>Corrected 10-25:</b> the cohort-level state drop is carried by <b>PS94</b> (chronic &minus;0.53, still falling); the other three lose only 0.01&ndash;0.09. So in 3 / 4 animals the state readout is nearly intact while position falls by half acutely &mdash; the specificity claim is, if anything, stronger. PS94's state loss is its own question (its behaviour also recovers slowest). Session means, cue window, engaged trials; pre = leave one session out (state pre includes June, n = 11). Source: e3 per-animal tables from the 13n sidecars.</div>
<figure id="fig3c"><img src="s_per_animal.png" alt="Per-animal template amplitude, captured variance and state decoder" loading="lazy"><figcaption><b class="h">Fig 3c &middot; per animal: pre-stroke template amplitude, variance it captures, and the frozen state decoder.</b> Coloured = animals, black = mean of animals. Right panel: PS94 alone falls (section 3); left / middle: the template (section 6).</figcaption></figure>''')

# ---- section 4: chronic stability update
rep('''A refit decoder is back (&minus;0.01, z &minus;0.4). The gap between refit and frozen beyond its pre-stroke size is &gt; 0 in 4 / 4 animals (0.05&ndash;0.17) but not significant under the animals &rarr; sessions &rarr; blocks bootstrap, so "reorganisation" stays provisional.</div>''',
    '''A refit decoder is back (&minus;0.01, z &minus;0.4). The gap between refit and frozen beyond its pre-stroke size is &gt; 0 in 4 / 4 animals (0.05&ndash;0.17) but not significant under the animals &rarr; sessions &rarr; blocks bootstrap, so "reorganisation" stays provisional.<br><b>Updated 10-25 with the October sessions</b> (the 10-07 table had dropped 2 chronic sessions per animal; now 12 / 12 / 8 / 11): frozen decoder chronic level / pre 1.04 / .85 / .83 / .90 and <b>flat in 4 / 4</b> (unchanged); refit 1.10 / 1.11 / 1.02 / .99; the reorganisation index G (.056 / .168 / .141 / .070) is <b>still rising in 3 / 4</b> (PS92, PS93, PS95; mixed-model log-day slope +.030, z 4.5, section 13) &mdash; but the June &rarr; August group below shows G of this size from two months of drift alone, so read it as "still drifting", not reorganisation beyond drift.</div>''')

# ---- section 6: template per animal + correction
rep('''<div class="tbl"><table>
<tr><th>post-cue, pooled (matched)</th><th>pre</th><th>acute</th><th>subacute</th><th>chronic</th></tr>
<tr><td>fitted amplitude a</td><td>.749</td><td>.286</td><td>.527</td><td>.745</td></tr>
<tr><td>shape gap vs pre (&Delta;)</td><td>&ndash;</td><td>+.192</td><td>+.128</td><td>+.186</td></tr>
<tr><td>variance captured by the template</td><td>61 %</td><td>18 %</td><td>36 %</td><td>41 %</td></tr>
</table></div>''', '''<h4>Per animal (added 10-25; session means, cue window, engaged; pre = leave one session out, August, n = 8)</h4>
<div class="tbl"><table>
<tr><th>animal</th><th>amplitude a: pre / acute / subacute / chronic</th><th>variance captured (frozen / ceiling)</th></tr>
<tr><td>PS92</td><td>.96 / .66 / .78 / 1.19</td><td>.71 / .32 / .39 / .45</td></tr>
<tr><td>PS93</td><td>.96 / .17 / .59 / .84</td><td>.73 / .11 / .37 / .38</td></tr>
<tr><td>PS94</td><td>.93 / .17 / .64 / .68</td><td>.66 / .11 / .42 / .47</td></tr>
<tr><td>PS95</td><td>.91 / .44 / .69 / .73</td><td>.56 / .06 / .30 / .24</td></tr>
</table></div>
<div class="tbl"><table>
<tr><th>post-cue, pooled (full-cohort sidecar, 10-07)</th><th>pre</th><th>acute</th><th>subacute</th><th>chronic</th></tr>
<tr><td>fitted amplitude a</td><td>.943</td><td>.361</td><td>.675</td><td>.883</td></tr>
<tr><td>variance captured (09-10 numbers)</td><td>61 %</td><td>18 %</td><td>36 %</td><td>41 %</td></tr>
</table></div>''')
rep('''<div class="take">The template's amplitude falls and fully returns; the shape mismatch beyond the pre-stroke cross-day cost does not recover.''',
    '''<div class="take"><b>Corrected 10-25 (per animal):</b> the amplitude falls in 4 / 4 but returns to pre only in PS92 / PS93 (PS94 .68, PS95 .73 of ~.93 still below) &mdash; "fully returns" was the pooled 09-10 three-animal number (.749 &rarr; .745). The variance the pre-stroke template captures stays below pre in <b>4 / 4</b>, so the shape claim stands. Mixed model (section 13): amplitude steps &minus;0.62 at the stroke (z &minus;3.8), recovers with log-day (+0.16, q .003) and tracks the deficit (z &minus;8.7).''')
rep('''Pooled only (no per-animal table yet); numbers pulled 09-10 with 3 chronic animals, figures rendered 10-07 on the full cohort.</div>''',
    '''Figures rendered 10-07 on the full cohort; per-animal panels in Fig 3c.</div>''')

# ---- section 10 note
rep('''<div class="take">Acutely the displacement is focal''',
    '''<p class="input">Re-run 10-25 with the October chronic sessions included: drift-consistent in 4 / 4, table unchanged.</p>
<div class="take">Acutely the displacement is focal''')

# ---- new section 13: mixed models (before "closed")
rep('''<section id="closed" data-toc="Closed, corrected, not established">''', '''<section id="lmm" data-toc="13. Mixed models" data-kind="n">
<span class="tag">Session-level mixed models &middot; re-run 10-25 on the rebuilt attempts table</span>
<h2>13. Mixed models: recovery with time and with behaviour <span class="kind n">neural</span><span class="st stands">stands</span></h2>
<p class="input">Per-session values, animal random effects (<code>scripts/imaging_lmm.py</code>). <b>time</b> model: step at the stroke + log(days) slope; <b>behaviour</b> model: slope against the behavioural deficit (default = time to first far-contra contact on LP-attempted trials; <code>--deficit contact</code> = contact-based). FDR within source; * = q &lt; .05. Re-run 10-25 by the FaceRhythm window on <code>trial_attempts.csv</code> 10-24 (complete cam1); the 10-21 tables used an attempts table with cam1 on 40 % / 0 % of PS94 / PS95 trials.</p>
<div class="tbl"><table class="wide">
<tr><th>measure</th><th>step at stroke</th><th>log-day slope</th><th>vs deficit (time-to-contact)</th><th>vs deficit (contact)</th><th>change vs 10-21</th></tr>
<tr><td>frozen decoder (matched, post only)</td><td>day-1 level .41</td><td>+.064 (z 3.9)*</td><td>&minus;.040 (z &minus;3.2, q .009)*</td><td>&minus;.276 (z &minus;4.5)*</td><td>same</td></tr>
<tr><td>refit decoder</td><td>day-1 level .51</td><td>+.094 (z 9.9)*</td><td>&minus;.059 (q .015)*</td><td>&minus;.377 (z &minus;4.0)*</td><td>same</td></tr>
<tr><td>reorganisation G (refit &minus; frozen)</td><td>&ndash;</td><td>+.030 (z 4.5)*</td><td>n.s. (q .16)</td><td>n.s. (q .054)</td><td>slope stronger (was z 2.2)</td></tr>
<tr><td>template amplitude (11amp)</td><td>&minus;.62 (z &minus;3.8)*</td><td>+.156 (q .003)*</td><td>&minus;.085 (z &minus;8.7)*</td><td>* </td><td>same</td></tr>
<tr><td>template captured (11c frozen)</td><td>&minus;.37 (z &minus;5.0)*</td><td>+.058 (q .010)*</td><td>z &minus;7.9*</td><td>q .018*</td><td>same</td></tr>
<tr><td>template ceiling (11c)</td><td>&minus;.16 (q .036)*</td><td>+.062 (q .009)*</td><td>n.s.</td><td>n.s.</td><td>same</td></tr>
<tr><td>best-match accuracy (10)</td><td>&minus;.40 (z &minus;8.7)*</td><td>+.082 (z 5.9)*</td><td>z &minus;7.6*</td><td>*</td><td>same</td></tr>
<tr><td>CD leave-one-out, far contra, pre-cue</td><td>&minus;.53 (z &minus;3.0, q .019)*</td><td>n.s.</td><td><b>n.s. (q .25)</b></td><td>q .050 (borderline)</td><td><b>was q .023</b></td></tr>
<tr><td>CD leave-one-out, far contra, cue</td><td>&minus;.23 (q .062)</td><td>n.s.</td><td>n.s.</td><td>&minus;.22 (q .043)*</td><td>&ndash;</td></tr>
<tr><td>position-coding ceiling (trend)</td><td>n.s.</td><td>+.016 (q .003)*</td><td><b>z &minus;4.0 (q &lt; .001)*</b></td><td>z &minus;4.9*</td><td><b>was q .080</b></td></tr>
</table></div>
<div class="take">Every decoder and template readout steps down at the stroke, recovers with log(time) and tracks the behavioural deficit session by session (the more impaired the session, the lower the code), under both deficit definitions. The reorganisation index G rises with time but does not track behaviour &mdash; consistent with drift (section 4's June &rarr; August group). Two rows moved with the rebuilt attempts table: the CD leave-one-out pre-cue projection no longer tracks the time-to-contact deficit (q .023 &rarr; .25; borderline with the contact deficit), and the position-coding ceiling now does (q .080 &rarr; &lt; .001). The pixel maps (section 7) are not in these models; they keep their bootstrap contours.</div>
<figure id="fig13"><img src="m_dec_time.png" alt="Frozen and refit decoder over days, mixed model" loading="lazy"><figcaption><b class="h">Fig 13 &middot; frozen / refit decoder (matched, post-stroke) over days, with the mixed-model fit.</b> Panels: frozen, refit, gap; F_deficit, G_reorg, refit deficit. Triangles = sessions coloured by animal; line = fit (animal random effects). Titles give the day-1 level and log-day slope.</figcaption></figure>
<figure id="fig13b"><img src="m_dec_beh.png" alt="Decoder against behavioural deficit" loading="lazy"><figcaption><b class="h">Fig 13b &middot; the same readouts against the behavioural deficit (time to far-contra contact).</b></figcaption></figure>
<figure id="fig13c"><img src="m_tmpl_time.png" alt="Template measures over days" loading="lazy"><figcaption><b class="h">Fig 13c &middot; pre-stroke template (family 11) over days.</b></figcaption></figure>
<figure id="fig13d"><img src="m_trend_time.png" alt="Position-coding trend over days" loading="lazy"><figcaption><b class="h">Fig 13d &middot; position-coding ceiling (trend family) over days.</b></figcaption></figure>
<figure id="fig13e"><img src="m_trend_beh.png" alt="Position-coding trend against deficit" loading="lazy"><figcaption><b class="h">Fig 13e &middot; position-coding ceiling against the deficit</b> (now significant, z &minus;4.0).</figcaption></figure>
</section>

<section id="closed" data-toc="Closed, corrected, not established">''')

# ---- corrected list + pending
rep('''<li><b>Corrected:</b> "frozen decoder back at recovery" (section 4);''',
    '''<li><b>Corrected 10-25:</b> the cohort state-decoder drop is PS94 alone (section 3); the template amplitude returns in 2 / 4, not "fully" (section 6); CD leave-one-out vs deficit no longer significant on the rebuilt attempts table (section 13).</li>
<li><b>Corrected:</b> "frozen decoder back at recovery" (section 4);''')
rep('''<li>Re-run the imaging mixed models with the rebuilt (complete) attempts table.</li>
<li>Failed-attempt decoder and maps: movement-matched control; statistics on the trial-class maps.</li>
<li>Per-animal tables for the state decoder and the pre-stroke template.</li>
<li>Re-run the chronic-stability decoders with the October sessions (a dropped-session bug was fixed).</li>''',
    '''<li>Failed-attempt decoder and maps: movement-matched control; statistics on the trial-class maps (encoding window).</li>
<li>Done 10-25: mixed models re-run on the rebuilt attempts table (section 13); per-animal state / template tables (sections 3, 6); chronic stability with the October sessions (sections 4, 10).</li>''')
p.write_text(s, encoding="utf-8")
print("ok")

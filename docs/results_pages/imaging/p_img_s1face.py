from pathlib import Path

p = Path(__file__).parent / "imaging_page" / "imaging_results.html"
s = p.read_text(encoding="utf-8")
a = '<div class="take">Re-fitting each day, position stays decodable in every epoch'
assert s.count(a) == 1
i = s.index("</div>", s.index(a)) + len("</div>")
s = s[:i] + ('\n<p class="input"><b>Cross-reference (10-25):</b> face movement alone decodes position in the same pre-cue window at a similar level '
             '(lick-free trials, 2 s window .43 pre, .34 with the last 0.5 s dropped, chance .17, 31 / 31 sessions; far contra drops to chance acutely &mdash; '
             'FR page section 6; FINDINGS 5.13), so the cortical pre-cue code may partly reflect face posture.</p>') + s[i:]
p.write_text(s, encoding="utf-8")
print("ok")

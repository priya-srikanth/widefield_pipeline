from pathlib import Path

p = Path(r"C:\Users\SabatiniLab\AppData\Local\Temp\claude\C--Users-SabatiniLab\c67bec01-3925-49ff-9cdc-8ff295df4bfa\scratchpad\imaging_page\imaging_results.html")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


# 1) collapsible figure groups in the TOC (same as the encoding page)
rep(".toc .sub a { color:var(--muted); font-size:.82rem }",
    ".toc .sub a { color:var(--muted); font-size:.82rem }\n"
    ".toc .grphead { display:flex; align-items:flex-start; gap:2px }\n"
    ".toc .grphead a { flex:1; min-width:0 }\n"
    ".toc .tg { flex:none; font:600 .7rem/1.4 \"Source Sans 3\", sans-serif; color:var(--muted); background:none; border:1px solid var(--rule); border-radius:3px; padding:1px 4px; margin-top:2px; cursor:pointer; min-width:30px }\n"
    ".toc .tg:hover { background:var(--tag) }\n"
    ".toc .sub2 { margin:2px 0 4px 14px }\n"
    ".toc .sub2 a { font-size:.78rem }")
rep("""      var ol = document.createElement('ol'); ol.className = 'sub';
      figs.forEach(function (f) {
        var b = f.querySelector('figcaption b.h');
        var txt = b ? b.textContent.replace(/[.:]$/, '') : f.id;
        var l2 = document.createElement('li'); var a2 = document.createElement('a'); a2.href = '#' + f.id; a2.textContent = txt;
        l2.appendChild(a2); ol.appendChild(l2);
      });
      li.appendChild(ol);""",
    """      // group by figure NUMBER (Fig 7, 7b ... 7n -> one collapsible "Fig 7" row), Priya 10-25
      var ol = document.createElement('ol'); ol.className = 'sub';
      var groups = [], byNum = {};
      figs.forEach(function (f) {
        var b = f.querySelector('figcaption b.h');
        var txt = b ? b.textContent.replace(/[.:]$/, '') : f.id;
        var m = /^fig(\\d+)/.exec(f.id); var num = m ? m[1] : f.id;
        if (!byNum[num]) { byNum[num] = []; groups.push(num); }
        byNum[num].push({ id: f.id, txt: txt });
      });
      groups.forEach(function (num) {
        var items = byNum[num];
        var g = document.createElement('li');
        var head = document.createElement('div'); head.className = 'grphead';
        var a1 = document.createElement('a'); a1.href = '#' + items[0].id; a1.textContent = items[0].txt;
        if (items.length > 1) {
          var key = 'tocgrp-' + num, open = false;
          try { open = localStorage.getItem(key) === '1'; } catch (e) {}
          var tg = document.createElement('button'); tg.type = 'button'; tg.className = 'tg';
          tg.setAttribute('aria-label', 'Show or hide figures ' + num + 'b onwards');
          var sub = document.createElement('ol'); sub.className = 'sub2'; sub.hidden = !open;
          tg.setAttribute('aria-expanded', String(open)); tg.textContent = (open ? '\\u25BE ' : '\\u25B8 ') + items.length;
          tg.addEventListener('click', function () {
            var o = sub.hidden; sub.hidden = !o; tg.setAttribute('aria-expanded', String(o));
            tg.textContent = (o ? '\\u25BE ' : '\\u25B8 ') + items.length;
            try { localStorage.setItem(key, o ? '1' : '0'); } catch (e) {}
          });
          items.slice(1).forEach(function (it) {
            var l3 = document.createElement('li'); var a3 = document.createElement('a'); a3.href = '#' + it.id; a3.textContent = it.txt;
            l3.appendChild(a3); sub.appendChild(l3);
          });
          head.appendChild(tg); head.appendChild(a1); g.appendChild(head); g.appendChild(sub);
        } else {
          head.appendChild(a1); g.appendChild(head);
        }
        ol.appendChild(g);
      });
      li.appendChild(ol);""")

# 2) every map shows its gate; LP twins of the contact-only maps
C = '<span class="st">contact</span> '
L = '<span class="st">LP lick</span> '
for fid, tag, twin in [("fig7", C, "7l"), ("fig7b", C, "7l"), ("fig7c", L, None), ("fig7d", L, "7f"),
                       ("fig7e", C, "7m"), ("fig7f", C, "7d"), ("fig7g", C, "7h"), ("fig7h", L, "7g"),
                       ("fig7i", C, "7n")]:
    a = '<figure id="%s">' % fid
    i = s.index(a)
    j = s.index("</figcaption>", i)
    k = s.index("<figcaption>", i) + len("<figcaption>")
    tw = (' <i>%s twin: Fig %s.</i>' % ("LP" if tag == C else "contact", twin)) if twin else ""
    s = s[:k] + tag + s[k:j] + tw + s[j:]

rep(" Fig 7k compares the stroke changes",
    " Every map is labelled with its gate (contact = DAQ spout contact; LP lick = Lightning Pose lick attempts) and names its twin under the other gate. ENL (Fig 7j) is defined on lick-free trials, so the gate barely matters there and it has no twin. Fig 7k compares the stroke changes")
anchor = "the LP, pre-cue-reference and ENL arms are rendering.</figcaption></figure>"
rep(anchor, anchor + "\n" + "\n".join([
    '<figure id="fig7l"><img src="p_lp_rest_cue.png" alt="Position maps, rest reference, cue window, LP lick gate" loading="lazy"><figcaption>' + L + '<b class="h">Fig 7l &middot; REST reference, cue window, LP lick gate (10-24).</b> Contact twin: Figs 7 / 7b.</figcaption></figure>',
    '<figure id="fig7m"><img src="p_lp_pre_cue.png" alt="Position maps, pre-cue reference, cue window, LP lick gate" loading="lazy"><figcaption>' + L + '<b class="h">Fig 7m &middot; PRECUE reference, cue window, LP lick gate (10-24).</b> Contact twin: Fig 7e.</figcaption></figure>',
    '<figure id="fig7n"><img src="p_lp_rest_precue.png" alt="Position maps, rest reference, pre-cue window, LP lick gate" loading="lazy"><figcaption>' + L + '<b class="h">Fig 7n &middot; REST reference, pre-cue window, LP lick gate (10-24).</b> Contact twin: Fig 7i.</figcaption></figure>',
]))
p.write_text(s, encoding="utf-8")
print("ok")

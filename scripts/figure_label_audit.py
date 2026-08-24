"""DOES EVERY FIGURE THAT VARIES BY A PARAMETER SAY SO ON THE FIGURE?

    PYTHONPATH=$(pwd) python scripts/figure_label_audit.py

THE BUG CLASS THIS EXISTS FOR (two instances found 2026-08-24, both by Priya reading slides):

  * `fig_confusion_alltrials` was called once per ARM but hardcoded "ALL trials" in its titles, so
    every LICK-ONLY figure was captioned as the all-trials one.
  * `fig_miss_vs_stopped` indexed x per CLASS, so two classes with different session sets were
    drawn misaligned and the extra points went unlabelled.

Both share a shape: a value that DISCRIMINATES one figure from another reaches the filename or the
data but not the label. Neither can fail a unit test -- the data is right -- and neither trips the
deck's figures-placed/missing counter, because the figure is produced and placed. The only detector
that has ever worked is a person noticing two things that should differ look the same.

WHAT THIS CHECKS, statically: for every function whose name starts with `fig`, if it takes a
discriminating parameter (align, arm, meth, method, cls, window, phase), does that parameter appear
anywhere inside an f-string in the body? A function that also accepts `suptitle` is exempt when the
caller supplies one, since the label then comes from outside -- those are reported separately rather
than as failures, because the check cannot see whether the caller actually included it.

FALSE POSITIVES ARE EXPECTED and are the point: this is a list to read, not a gate. A parameter that
only selects data and is genuinely irrelevant to the reader (a filename stem, a dpi) shows up here
and can be dismissed in one glance -- which is much cheaper than the alternative.
"""
import ast
import re
from pathlib import Path

MODULES = sorted(Path("wfield_local").glob("*.py"))
DISCRIMINATING = {"align", "arm", "arm_name", "meth", "method", "cls", "window", "phase", "position"}


def fstring_text(node):
    """Every literal fragment of every f-string and string in a subtree, joined."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.JoinedStr):
            for v in n.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    out.append(ast.unparse(v.value))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return " ".join(out)


def titles_of(fn):
    """Text passed to suptitle/set_title/title, WITH local variables resolved one level.

    Titles are routinely assembled first (`ttl = f"... {arm_name} ..."`) and then passed
    (`ax.set_title(f"{an} - {ttl}")`), so scanning only the call site misses the parameter and
    reports a false positive -- which is exactly what the first run of this script did for the
    function it was written to catch. Any local whose name appears in a title has its own assigned
    text folded in.
    """
    assigned = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = fstring_text(n.value)
    # DERIVED NAMES. A parameter is routinely renamed for display before it reaches a title --
    # `disp = dict(ALIGNS)[align]`, then f"... {disp} window ...". Looking for the literal token
    # `align` in the title then reports a false positive, which on the first broad run flagged
    # eleven functions that in fact label themselves correctly. Propagate to a fixed point so
    # multi-step renames are covered too.
    derived = {p: {p} for p in DISCRIMINATING}

    def pairs(target, value):
        """(target-name, value-node) pairs, unpacking tuple assignment element-wise.

        `disp, R = dict(ALIGNS)[align], res["methods"][meth]` is one Assign with a Tuple target.
        Handling only plain Name targets skipped it and kept reporting the function as unlabelled --
        the third time this checker missed an indirection, which is itself the point: this bug class
        hides behind one more hop than whatever you last accounted for.
        """
        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple) \
                and len(target.elts) == len(value.elts):
            for t, v in zip(target.elts, value.elts):
                yield from pairs(t, v)
        elif isinstance(target, ast.Name):
            yield target.id, value
        elif isinstance(target, ast.Tuple):
            for t in target.elts:               # can't pair up: every element sees every source
                yield from pairs(t, value)

    for _ in range(4):
        for n in ast.walk(fn):
            if not isinstance(n, ast.Assign):
                continue
            for tgt in n.targets:
                for name, val in pairs(tgt, n.value):
                    names = {x.id for x in ast.walk(val) if isinstance(x, ast.Name)}
                    for group in derived.values():
                        if names & group:
                            group.add(name)
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in ("suptitle", "set_title", "title", "set_xlabel", "set_ylabel"):
                txt = fstring_text(n)
                # WORD BOUNDARIES, not `in`. A substring test folds the text of a local named `p`
                # into any title containing the letter p -- which on the first run pulled a
                # FILENAME's `{align}` into a title that never mentions the alignment, turning a
                # true positive into a false negative. The checker had the bug it was written to
                # find.
                for var, vtxt in assigned.items():
                    if re.search(rf"\b{re.escape(var)}\b", txt):
                        txt += " " + vtxt
                out.append(txt)
    return " ".join(out), derived


def mentions(param, title_text, derived):
    """Does the title name this parameter, or any local derived from it?"""
    return any(re.search(rf"\b{re.escape(alias)}\b", title_text)
               for alias in derived.get(param, {param}))


rows = []
for mod in MODULES:
    try:
        tree = ast.parse(mod.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if not fn.name.startswith("fig"):
            continue
        params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        # A PARAMETER THE BODY NEVER READS CANNOT DISCRIMINATE ANYTHING, so it does not need to be
        # labelled. `figure_engagement(res, out, align, meth)` takes `meth` only to match the
        # uniform signature the caller dispatches on -- the behaviour panel is method-independent,
        # which is exactly why its filename carries no method either.
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        disc = sorted(params & DISCRIMINATING & used)
        if not disc:
            continue
        title_text, derived = titles_of(fn)
        missing = [d for d in disc if not mentions(d, title_text, derived)]
        rows.append((mod.name, fn.name, fn.lineno, disc, missing, "suptitle" in params))

print(f"{len(rows)} figure functions take a discriminating parameter\n")
bad = [r for r in rows if r[4] and not r[5]]
soft = [r for r in rows if r[4] and r[5]]
ok = [r for r in rows if not r[4]]

print(f"--- {len(bad)} DO NOT reference it in any title, and take no caller suptitle")
for m, f, ln, disc, missing, _ in bad:
    print(f"  {m}:{ln} {f}()  takes {disc}, never mentions {missing}")
print(f"\n--- {len(soft)} do not reference it but DO accept `suptitle` (caller may supply it)")
for m, f, ln, disc, missing, _ in soft:
    print(f"  {m}:{ln} {f}()  takes {disc}, never mentions {missing}")
print(f"\n--- {len(ok)} name every discriminating parameter in a title")
for m, f, ln, disc, _missing, _ in ok:
    print(f"  {m}:{ln} {f}()  {disc}")

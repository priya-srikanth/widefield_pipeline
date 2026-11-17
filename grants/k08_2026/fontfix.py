"""Rename embedded Liberation Sans (metric clone of Arial, used by the renderer here) to Arial's
PostScript names, so Illustrator maps the text to the installed Arial instead of flagging a
missing font."""
import sys, pikepdf
MAP = {'LiberationSans-Bold': 'Arial-BoldMT', 'LiberationSans-Italic': 'Arial-ItalicMT',
       'LiberationSans-BoldItalic': 'Arial-BoldItalicMT', 'LiberationSans': 'ArialMT'}
def fix(path):
    p = pikepdf.open(path, allow_overwriting_input=True); n = 0
    for obj in p.objects:
        if not isinstance(obj, pikepdf.Dictionary): continue
        for key in ('/BaseFont', '/FontName'):
            if key in obj:
                v = str(obj[key]).lstrip('/')
                pre, _, base = v.partition('+') if '+' in v else ('', '', v)
                if base in MAP:
                    obj[key] = pikepdf.Name('/' + (pre + '+' if pre else '') + MAP[base]); n += 1
        if obj.get('/FamilyName') is not None and 'Liberation' in str(obj.FamilyName):
            obj.FamilyName = pikepdf.String('Arial')
    p.save(path); print(path, n)
for f in sys.argv[1:]: fix(f)

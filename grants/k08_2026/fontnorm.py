"""Every text element in the composed figures set to Arial (sources carried DejaVu, Liberation,
Inter and unstyled serif text), so Illustrator finds the font and nothing is substituted."""
import re, sys
for f in sys.argv[1:]:
    s = open(f, encoding='utf-8').read()
    s = re.sub(r"font-family:\s*[^;\"]+", "font-family: Arial", s)
    s = re.sub(r'font-family="[^"]*"', 'font-family="Arial"', s)
    if '<style id="arial-all">' not in s:
        s = re.sub(r'(<svg[^>]*>)', r'\1\n<style id="arial-all">text, tspan { font-family: Arial; }</style>', s, count=1)
    open(f, 'w', encoding='utf-8').write(s); print('normalised', f)

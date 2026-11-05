"""Append titled full-slide images to an existing .pptx (used to add the epileptiform-discharge screen
and the 490-vs-470 LED panels to the cohort2 preprocessing deck after preprocess_deck.py builds it).

  python scripts/cohort2_qc/append_qc_slides.py <deck.pptx> "Title::/path/fig.png" ["Title2::/path2.png" ...]
Missing images are skipped with a note. Writes the deck in place."""
import sys, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def png_size(p):
    import struct
    with open(p,"rb") as f:
        f.read(16); w,h=struct.unpack(">II",f.read(8)); return w,h

def main():
    deck=sys.argv[1]; items=sys.argv[2:]
    prs=Presentation(deck); SW,SH=prs.slide_width,prs.slide_height
    blank=prs.slide_layouts[6]
    for it in items:
        title,_,img=it.partition("::")
        if not os.path.exists(img): print(f"  skip (missing): {img}"); continue
        s=prs.slides.add_slide(blank)
        # dark bg
        from lxml import etree
        ns="{http://schemas.openxmlformats.org/drawingml/2006/main}"
        cSld=s._element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}cSld")
        bg=etree.SubElement(cSld,"{http://schemas.openxmlformats.org/presentationml/2006/main}bg")
        bgPr=etree.SubElement(bg,"{http://schemas.openxmlformats.org/presentationml/2006/main}bgPr")
        sf=etree.SubElement(bgPr,f"{ns}solidFill"); c=etree.SubElement(sf,f"{ns}srgbClr"); c.set("val","222222")
        etree.SubElement(bgPr,f"{ns}effectLst")
        tb=s.shapes.add_textbox(Inches(0.3),Inches(0.1),SW-Inches(0.6),Inches(0.6)).text_frame
        tb.text=title; tb.paragraphs[0].runs[0].font.size=Pt(20); tb.paragraphs[0].runs[0].font.bold=True
        tb.paragraphs[0].runs[0].font.color.rgb=RGBColor(0xEE,0xEE,0xEE)
        w,h=png_size(img); boxw=SW-Inches(0.6); boxh=SH-Inches(1.0)
        sc=min(boxw/w,boxh/h); iw,ih=int(w*sc),int(h*sc)
        s.shapes.add_picture(img,int((SW-iw)/2),Inches(0.8),iw,ih)
        print(f"  added slide: {title}")
    prs.save(deck); print(f"saved {deck}")

if __name__=="__main__": main()

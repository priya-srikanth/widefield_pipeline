import sys, asyncio, pathlib
from playwright.async_api import async_playwright
async def main(src, out, w, h, scale=2):
    html = '<html><body style="margin:0"><img src="file://%s" width="%d" height="%d"></body></html>' % (
        pathlib.Path(src).resolve(), w, h)
    p = pathlib.Path('/tmp/_rast.html'); p.write_text(html)
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page(viewport={'width': w, 'height': h}, device_scale_factor=scale)
        await pg.goto('file://' + str(p)); await pg.wait_for_timeout(1500)
        await pg.screenshot(path=out, full_page=True)
        await b.close()
    print('wrote', out, 'scale', scale)
asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))

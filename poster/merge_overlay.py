from pathlib import Path

from pypdf import PdfReader, PdfWriter


root = Path(__file__).resolve().parent
source = PdfReader(root / "poster_source.pdf")
overlay = PdfReader(root / "poster_overlay.pdf")

page = source.pages[0]
page.merge_page(overlay.pages[0], over=True)

writer = PdfWriter()
writer.add_page(page)
with (root / "poster_vector.pdf").open("wb") as stream:
    writer.write(stream)

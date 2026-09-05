"""Create the delivery PDF as one high-resolution image-backed page.

The editable poster is assembled from a source PDF plus overlays. Flattening only the
delivery copy removes covered text and clipped copies of the old page that PDF viewers
would otherwise still parse. ``poster_vector.pdf`` remains available for future edits.
"""

from pathlib import Path
import subprocess
import tempfile

from pypdf import PdfReader
from reportlab.pdfgen import canvas


root = Path(__file__).resolve().parent
source = root / "poster_vector.pdf"
output = root / "poster.pdf"

page = PdfReader(source).pages[0]
width = float(page.mediabox.width)
height = float(page.mediabox.height)

with tempfile.TemporaryDirectory(prefix="poster-flatten-") as tmp:
    image_prefix = Path(tmp) / "poster"
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "1",
            "-singlefile",
            "-r",
            "300",
            "-jpeg",
            "-jpegopt",
            "quality=96,optimize=y",
            str(source),
            str(image_prefix),
        ],
        check=True,
    )
    image_path = image_prefix.with_suffix(".jpg")
    temp_pdf = Path(tmp) / "poster.pdf"
    pdf = canvas.Canvas(str(temp_pdf), pagesize=(width, height), pageCompression=1)
    pdf.drawImage(str(image_path), 0, 0, width=width, height=height)
    pdf.showPage()
    pdf.save()
    temp_pdf.replace(output)

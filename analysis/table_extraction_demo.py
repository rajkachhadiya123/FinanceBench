"""
Demo (not pipeline code): compare how a financial table looks when extracted
as PLAIN TEXT (PyMuPDF) vs as a STRUCTURED TABLE (pdfplumber), on one real
page. Also checks whether that page (or others) contain embedded images
(charts/graphs), since those are handled completely differently.
"""
from pathlib import Path

import fitz
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "RAW" / "3M_2018_10K.pdf"
PAGE_NUM = 59  # Consolidated Statement of Cash Flows (0-indexed)


def safe(t: str) -> str:
    return t.encode("ascii", errors="replace").decode("ascii")


def show_plain_text():
    doc = fitz.open(PDF_PATH)
    text = doc[PAGE_NUM].get_text()
    doc.close()
    print("=" * 70)
    print("1) RAW TEXT (PyMuPDF get_text) -- first 900 chars")
    print("=" * 70)
    print(safe(text[:900]))
    print()


def show_structured_table():
    print("=" * 70)
    print("2) STRUCTURED TABLE (pdfplumber extract_table)")
    print("=" * 70)
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PAGE_NUM]
        tables = page.extract_tables()
        print(f"Number of tables pdfplumber detected on this page: {len(tables)}")
        if tables:
            table = tables[0]
            print(f"Rows in first table: {len(table)}")
            print("\nFirst 8 rows as raw cell lists:")
            for row in table[:8]:
                print(" ", row)

            print("\nSame rows converted to Markdown table:")
            for i, row in enumerate(table[:8]):
                cells = [str(c).replace("\n", " ") if c else "" for c in row]
                print("| " + " | ".join(cells) + " |")
                if i == 0:
                    print("|" + "---|" * len(cells))


def check_for_images_and_charts():
    print("\n" + "=" * 70)
    print("3) CHECKING FOR EMBEDDED IMAGES/CHARTS ACROSS THE DOCUMENT")
    print("=" * 70)
    doc = fitz.open(PDF_PATH)
    pages_with_images = []
    for i in range(doc.page_count):
        images = doc[i].get_images()
        if images:
            pages_with_images.append((i, len(images)))
    print(f"Total pages: {doc.page_count}")
    print(f"Pages containing at least one embedded image: {len(pages_with_images)}")
    print("First 10 such pages (page_index, image_count):", pages_with_images[:10])

    if pages_with_images:
        sample_page_idx = pages_with_images[0][0]
        print(f"\nText PyMuPDF extracts from page {sample_page_idx} (the one with an image):")
        print(safe(doc[sample_page_idx].get_text()[:500]))
    doc.close()


if __name__ == "__main__":
    show_plain_text()
    show_structured_table()
    check_for_images_and_charts()

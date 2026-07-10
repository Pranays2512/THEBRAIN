#!/usr/bin/env python3
"""ocr_pdf.py — the brain's EYES: rasterize each page and OCR it to text.

extract_to_txt.py works only when the PDF has a real text layer with a Unicode
map. NCERT-Class-10-Science.pdf uses a custom (non-Unicode) font, so get_text
yields glyph garbage (109 English words in 780K chars). This renders each page
to a bitmap and runs tesseract over it instead — the same thing a human eye does.

    ../venv2/bin/python3 ocr_pdf.py <in.pdf> <out.txt> [dpi]
"""
import os
import re
import sys
import time

os.environ.setdefault("TESSDATA_PREFIX", "/opt/homebrew/opt/tesseract/share/tessdata")

import fitz  # PyMuPDF


def ocr(pdf_path, out_path, dpi=200):
    doc = fitz.open(pdf_path)
    n = len(doc)
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            page = doc[i]
            tp = page.get_textpage_ocr(flags=0, full=True, dpi=dpi)
            txt = page.get_text(textpage=tp)
            f.write(txt)
            f.write("\n")
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"  OCR {i + 1}/{n} pages  ({time.time() - t0:.0f}s)", flush=True)
    doc.close()
    body = open(out_path, errors="ignore").read()
    words = len(re.findall(r"[A-Za-z]{3,}", body))
    print(f"done: {n} pages -> {out_path}  ({words} English words, {time.time() - t0:.0f}s)",
          flush=True)
    return words


if __name__ == "__main__":
    ocr(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)

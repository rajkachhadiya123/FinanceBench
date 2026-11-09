# analysis

Early exploration scripts, kept for reference. They are not part of the pipeline.

| Script | Purpose | Finding |
|---|---|---|
| `pdf_extraction_test.py` | Checks whether PyMuPDF's plain-text extraction contains FinanceBench's evidence text | Plain extraction captures the evidence content; evidence is usually a short excerpt of a page, so containment (not equality) is the right check |
| `table_extraction_demo.py` | Compares one financial table as plain text (PyMuPDF) vs a structured table (pdfplumber), and looks for embedded charts | Plain text was kept for the pipeline; a later table-to-Markdown attempt was dropped because it fragmented tables and split words (see the experiment log). No benchmark question depends on a chart |

`table_extraction_demo.py` needs `pdfplumber`, which is not in `requirements.txt`.

# -*- coding: utf-8 -*-
import glob, os
import pdfplumber

files = [f for f in glob.glob(r"C:\Users\joon2\Desktop\*.pdf")]
print("found:", files)
out = r"C:\Users\joon2\Desktop\da2\artifacts\chatgpt_playbook.txt"
with pdfplumber.open(files[0]) as pdf, open(out, "w", encoding="utf-8") as o:
    print("pages:", len(pdf.pages))
    for i, page in enumerate(pdf.pages, 1):
        t = page.extract_text() or ""
        o.write(f"\n===== PAGE {i} =====\n{t}\n")
print("saved ->", out)

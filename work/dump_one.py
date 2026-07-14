# -*- coding: utf-8 -*-
import json, os
DATA = r"C:\Users\joon2\Desktop\da2\open\data"
OUT = r"C:\Users\joon2\Desktop\da2\artifacts\one_record.txt"
lab = {}
import csv
with open(os.path.join(DATA, "train_labels.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lab[row["id"]] = row["action"]
pick = None
with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        # a compact but complete example: 2 history turns, korean
        if r["session_meta"].get("language_pref") == "ko" and len(r.get("history") or []) == 4:
            pick = r
            break
with open(OUT, "w", encoding="utf-8") as o:
    o.write("LABEL (정답, train_labels.csv에 별도 저장): " + lab[pick["id"]] + "\n\n")
    o.write("=== train.jsonl 한 줄 = 한 샘플 (아래는 보기 좋게 편집) ===\n\n")
    o.write(json.dumps(pick, ensure_ascii=False, indent=2))
print("saved", OUT)

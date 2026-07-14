# -*- coding: utf-8 -*-
"""How out-of-domain is our text for XLM-R? Measure UNK rate, fertility
(subwords per word), and composition. Grounds the 'is the text special?' claim."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from transformers import AutoTokenizer
from featurize import build_text

DATA = r"C:\Users\joon2\Desktop\da2\open\data"
tok = AutoTokenizer.from_pretrained(r"C:\Users\joon2\Desktop\da2\submit\model\m0")
unk_id = tok.unk_token_id

recs = []
with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 6000: break
        recs.append(json.loads(line))

prompts = [r.get("current_prompt", "") or "" for r in recs]
serials = [build_text(r) for r in recs]

def stats(texts, name):
    n_unk = n_tok = n_char = n_word = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_unk += sum(1 for i in ids if i == unk_id)
        n_tok += len(ids)
        n_char += len(t)
        n_word += len(t.split())
    print(f"\n[{name}]  ({len(texts)} texts)")
    print(f"  UNK rate       : {n_unk/max(n_tok,1)*100:.3f}%   ({n_unk} unk / {n_tok} tokens)")
    print(f"  chars / token  : {n_char/max(n_tok,1):.2f}   (Korean 효율; 낮을수록 잘게 쪼갬)")
    print(f"  tokens / word  : {n_tok/max(n_word,1):.2f}   (fertility; 높을수록 낯섦)")

stats(prompts, "current_prompt only")
stats(serials, "full serialized (prompt+meta+hist)")

# composition of prompt text
allp = " ".join(prompts)
ko = len(re.findall(r"[가-힣]", allp))
en = len(re.findall(r"[A-Za-z]", allp))
dig = len(re.findall(r"[0-9]", allp))
sym = len(re.findall(r"[^\sA-Za-z0-9가-힣]", allp))
tot = ko+en+dig+sym
print(f"\n[prompt char composition]  한글 {ko/tot*100:.0f}% / 영문 {en/tot*100:.0f}% / 숫자 {dig/tot*100:.0f}% / 기호 {sym/tot*100:.0f}%")

# show 2 example tokenizations
print("\n[예시 토큰화 (한글 프롬프트가 어떻게 쪼개지나)]")
for p in prompts[:3]:
    if re.search(r"[가-힣]", p):
        toks = tok.tokenize(p)
        print(f"  원문: {p[:50]}")
        print(f"  토큰({len(toks)}): {toks[:20]}")
        break

# -*- coding: utf-8 -*-
"""Proof: vocab-prune the fine-tuned xlm-roberta-base fold-0 checkpoint.

Steps
  1. build_text() over train.jsonl + test.jsonl (identical serialization to training)
  2. tokenize (truncation 512) -> distinct token ids used
  3. holdout OOV estimate: mine on 80% of train sessions, measure token OOV on other 20%
  4. surgery: edit tokenizer.json (Unigram vocab list) + slice embedding rows, save fp16
  5. equivalence: original vs pruned logits on 256 train texts + OOV stress strings
  6. report sizes
"""
import json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\joon2\Desktop\da2\submit")
from featurize import build_text

DACON = r"C:\Users\joon2\Desktop\da2"
M0 = os.path.join(DACON, "submit", "model", "m0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m0_pruned")
MAXLEN = 512

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

t0 = time.time()
train = load_jsonl(os.path.join(DACON, "data", "train.jsonl"))
test = load_jsonl(os.path.join(DACON, "data", "test.jsonl"))
print(f"train={len(train)} test={len(test)}  ({time.time()-t0:.0f}s)", flush=True)

train_texts = [build_text(s) for s in train]
test_texts = [build_text(s) for s in test]
print(f"featurized ({time.time()-t0:.0f}s)", flush=True)

tok = AutoTokenizer.from_pretrained(M0)

def used_ids(texts, bs=2048):
    used = set()
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=MAXLEN)["input_ids"]
        for ids in enc:
            used.update(ids)
    return used

# session-wise 80/20 split for OOV estimate
sess = {}
for i, s in enumerate(train):
    sess.setdefault(s.get("session_meta", {}).get("session_id", s.get("id", i)), []).append(i)
keys = sorted(sess.keys()); random.Random(0).shuffle(keys)
cut = int(len(keys) * 0.8)
tr_idx = [i for k in keys[:cut] for i in sess[k]]
ho_idx = [i for k in keys[cut:] for i in sess[k]]

used_tr80 = used_ids([train_texts[i] for i in tr_idx])
print(f"distinct ids in 80% train: {len(used_tr80)}  ({time.time()-t0:.0f}s)", flush=True)

# holdout OOV rate (token occurrences whose id not in mined set)
oov_tok = tot_tok = 0; oov_rows = 0
for i in range(0, len(ho_idx), 2048):
    enc = tok([train_texts[j] for j in ho_idx[i:i+2048]], truncation=True, max_length=MAXLEN)["input_ids"]
    for ids in enc:
        miss = sum(1 for t in ids if t not in used_tr80)
        oov_tok += miss; tot_tok += len(ids); oov_rows += (miss > 0)
print(f"holdout 20%: token OOV {oov_tok}/{tot_tok} = {oov_tok/tot_tok*100:.4f}%  rows w/ OOV {oov_rows}/{len(ho_idx)} = {oov_rows/len(ho_idx)*100:.2f}%", flush=True)

used_train = used_tr80 | used_ids([train_texts[i] for i in ho_idx])
used_test = used_ids(test_texts)
print(f"distinct: train={len(used_train)} test={len(used_test)} test-not-in-train={len(used_test-used_train)}", flush=True)

# kept set: train+test + specials + all single-char pieces (safety margin for hidden test)
vocab_size = tok.vocab_size  # 250002
keep = set(used_train) | set(used_test) | {0, 1, 2, 3}
mask_id = tok.mask_token_id
keep.add(mask_id)
tj = json.load(open(os.path.join(M0, "tokenizer.json"), encoding="utf-8"))
vlist = tj["model"]["vocab"]  # [[piece, score], ...] index == id
print(f"tokenizer.json model.vocab len={len(vlist)} type={tj['model']['type']} unk_id={tj['model'].get('unk_id')}", flush=True)
n_singles = 0
for i, (piece, score) in enumerate(vlist):
    core = piece[1:] if piece.startswith("▁") else piece
    if len(core) <= 1:
        if i not in keep:
            n_singles += 1
        keep.add(i)
print(f"single-char margin added {n_singles}; kept total {len(keep)} / {len(vlist)}", flush=True)

kept_sorted = sorted(i for i in keep if i < len(vlist))
old2new = {o: n for n, o in enumerate(kept_sorted)}

# --- tokenizer.json surgery (direct file edit; format stays as-is for old tokenizers libs)
tj["model"]["vocab"] = [vlist[o] for o in kept_sorted]
tj["model"]["unk_id"] = old2new[3]
for at in tj.get("added_tokens", []):
    if at["id"] in old2new:
        at["id"] = old2new[at["id"]]
    else:  # token beyond pruned vocab (shouldn't happen; mask kept)
        at["id"] = old2new[mask_id]
# post_processor may reference special token ids
def fix_ids(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("ids", "tokens") and isinstance(v, list) and all(isinstance(x, int) for x in v):
                obj[k] = [old2new.get(x, x) for x in v]
            else:
                fix_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            fix_ids(v)
if tj.get("post_processor"):
    for sp in (tj["post_processor"].get("special_tokens") or {}).values():
        if isinstance(sp, dict) and isinstance(sp.get("ids"), list):
            sp["ids"] = [old2new.get(x, x) for x in sp["ids"]]

os.makedirs(OUT, exist_ok=True)
json.dump(tj, open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
for f in ("tokenizer_config.json",):
    src = os.path.join(M0, f)
    if os.path.exists(src):
        open(os.path.join(OUT, f), "w", encoding="utf-8").write(open(src, encoding="utf-8").read())

# --- model surgery
model = AutoModelForSequenceClassification.from_pretrained(M0)
emb = model.get_input_embeddings().weight.data
idx = torch.tensor(kept_sorted, dtype=torch.long)
new_emb = torch.nn.Embedding(len(kept_sorted), emb.shape[1], padding_idx=old2new[1])
new_emb.weight.data = emb[idx].clone()
model.set_input_embeddings(new_emb)
model.config.vocab_size = len(kept_sorted)
model.config.pad_token_id = old2new[1]
model.half()
model.save_pretrained(OUT, safe_serialization=True)
print(f"saved pruned model: vocab {len(vlist)} -> {len(kept_sorted)}", flush=True)

sz = lambda p: os.path.getsize(p) / 1e6
print(f"SIZE model.safetensors: {sz(os.path.join(M0,'model.safetensors')):.1f}MB -> {sz(os.path.join(OUT,'model.safetensors')):.1f}MB", flush=True)
print(f"SIZE tokenizer.json:    {sz(os.path.join(M0,'tokenizer.json')):.1f}MB -> {sz(os.path.join(OUT,'tokenizer.json')):.1f}MB", flush=True)

# --- equivalence check
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok2 = AutoTokenizer.from_pretrained(OUT)
m_orig = AutoModelForSequenceClassification.from_pretrained(M0).to(dev).eval()
m_new = AutoModelForSequenceClassification.from_pretrained(OUT).to(dev).eval()
if dev == "cuda":
    m_orig.half(); m_new.half()

rng = random.Random(1)
sample = [train_texts[i] for i in rng.sample(range(len(train_texts)), 256)]
stress = ["Zażółć gęślą jaźń Привет мир नमस्ते দুনিয়া ⚗️🜍 quixotic zephyrs",
          "ﷺ ᚠᚢᚦᚨᚱᚲ ⵣⵓⵔ 中文测试汉字罕见字 龘齉爨 ★unseen★"]
maxdiff = 0.0; agree = 0
with torch.no_grad():
    for i in range(0, len(sample), 64):
        b = sample[i:i+64]
        e1 = tok(b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        e2 = tok2(b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        assert e1["input_ids"].shape == e2["input_ids"].shape, "tokenization length changed!"
        l1 = m_orig(**e1).logits.float(); l2 = m_new(**e2).logits.float()
        maxdiff = max(maxdiff, (l1 - l2).abs().max().item())
        agree += (l1.argmax(-1) == l2.argmax(-1)).sum().item()
    print(f"EQUIVALENCE on 256 train texts: max|logit diff|={maxdiff:.2e}  argmax agree {agree}/256", flush=True)
    for s in stress:
        e2 = tok2([s], truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        l2 = m_new(**e2).logits.float()
        print(f"stress OK (no crash), ids={e2['input_ids'].shape[1]}, unk_frac={(e2['input_ids']==old2new[3]).float().mean().item():.2f}", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)

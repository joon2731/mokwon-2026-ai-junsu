# -*- coding: utf-8 -*-
"""경량 규칙으로 라벨 노이즈율 측정: '프롬프트가 명백히 X를 시키는데 라벨이 딴것'인 비율.

딥러닝 불사용 — 정규식 + 카운팅. sim/au 분리 측정 후 테스트(30k) 기대치 이식.
규칙은 정밀도 우선(보수적)으로 설계 — 애매하면 매칭 안 함.
"""
import csv
import io
import json
import re

ROOT = r"C:\Users\joon2\Desktop\da2"

labels = {}
for row in csv.DictReader(open(ROOT + r"\open\data\train_labels.csv", encoding="utf-8")):
    labels[row["id"]] = row["action"]

RULES = [
    ("run_tests", re.compile(
        r"(다시\s*테스트|테스트\s*(다시|재실행|돌려|실행해)|재테스트"
        r"|run\s+(the\s+)?tests?\b|rerun\s+(the\s+)?tests?|pytest\s*(돌려|run|실행)?$"
        r"|테스트\s*돌리)", re.I)),
    ("lint_or_typecheck", re.compile(
        r"(다시\s*(린트|타입체크)|린트\s*(돌려|다시|실행)|타입\s*체크\s*(돌려|다시|해)"
        r"|\b(run\s+)?(eslint|tsc|typecheck|lint)\b.{0,12}(돌려|다시|run|실행|check)?"
        r"|타입체크)", re.I)),
    ("respond_only", re.compile(
        r"(지금까지|여태|오늘)\s*(한|진행한|작업한).{0,10}(정리|요약)"
        r"|(정리|요약)(해|좀|부탁|해줄|해줘)|마무리하고.{0,8}(정리|요약)", re.I)),
    ("plan_task", re.compile(
        r"(계획\s*(세워|짜|수립)|단계(로|별로)?\s*(쪼개|나눠|정리해)"
        r"|플랜\s*(짜|세워)|작업\s*(순서|계획)\s*(정해|짜)|roadmap|plan\s+(out|the\s+steps))", re.I)),
    ("web_search", re.compile(
        r"(웹|인터넷|구글|공식\s*문서|docs?에서)\s*(에서)?\s*(검색|찾아|알아봐)"
        r"|search\s+(the\s+)?(web|online|docs)|look\s+up\s+online|검색해\s*봐", re.I)),
    ("list_directory", re.compile(
        r"(폴더|디렉토리|디렉터리)\s*(구조|목록|내용)\s*(보여|확인|봐)"
        r"|뭐(가|들)?\s*있(는지|나)\s*(좀\s*)?(봐|확인|보여)"
        r"|\bls\b|list\s+(the\s+)?(files|directory)", re.I)),
    ("run_bash", re.compile(
        r"(다시\s*빌드|재빌드|빌드\s*(다시|돌려|해봐|실행)"
        r"|rebuild|run\s+(the\s+)?(build|script)\b|스크립트\s*(돌려|실행))", re.I)),
]

tot = {"sim": 0, "au": 0}
stats = {}
examples = {}
for line in io.open(ROOT + r"\open\data\train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    src = "au" if r["id"].startswith("sess_au_") else "sim"
    tot[src] += 1
    p = r.get("current_prompt") or ""
    y = labels.get(r["id"])
    for cls, pat in RULES:
        if pat.search(p):
            key = (cls, src)
            n, ok = stats.get(key, (0, 0))
            hit = y == cls
            stats[key] = (n + 1, ok + int(hit))
            if not hit and len(examples.get(cls, [])) < 3 and src == "sim":
                examples.setdefault(cls, []).append((p[:60], y))
            break  # 첫 매칭 규칙만

print(f"{'규칙(의도)':20s} {'소스':4s} {'매칭':>6s} {'라벨일치':>8s} {'배신율':>7s}")
grand = {"sim": [0, 0], "au": [0, 0]}
for cls, _ in RULES:
    for src in ("sim", "au"):
        n, ok = stats.get((cls, src), (0, 0))
        if n == 0:
            continue
        grand[src][0] += n
        grand[src][1] += ok
        print(f"{cls:20s} {src:4s} {n:6d} {ok:8d} {100*(1-ok/n):6.1f}%")
print("-" * 55)
for src in ("sim", "au"):
    n, ok = grand[src]
    if n:
        print(f"{'전체':20s} {src:4s} {n:6d} {ok:8d} {100*(1-ok/n):6.1f}%  "
              f"(전체 {src} 중 매칭 {100*n/tot[src]:.1f}%)")

n, ok = grand["sim"]
if n:
    rate = 1 - ok / n
    cover = n / tot["sim"]
    est = 30000 * 0.85 * cover * rate  # 테스트 sim 비중 ~85% 가정
    print(f"\n테스트 이식 추정: 30k × sim85% × 매칭율 {cover:.1%} × 배신율 {rate:.1%} "
          f"≈ '명백한데 딴 라벨'인 행 ~{est:.0f}개")

print("\n[sim 배신 예시]")
for cls, exs in examples.items():
    for p, y in exs:
        print(f"  ({cls}) \"{p}\" -> 라벨 {y}")

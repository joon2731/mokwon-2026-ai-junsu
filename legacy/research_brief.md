# Deep Research Request: Maximizing Macro-F1 on Agent Next-Action Prediction

You are an expert ML-competition researcher. Produce a **deep, cited, actionable research report** to help me squeeze the maximum **macro-F1** on a hard text-classification task under tight deployment constraints. Prioritize **concrete techniques with evidence of real gains**, search recent (2023–2026) arXiv papers **and Kaggle/DACON winning-solution writeups**, and flag which techniques fit my constraints. You may write the final report in Korean or English.

## The task
Predict an AI coding agent's **next action** (single-label, 14 classes): `read_file, grep_search, glob_pattern, list_directory, edit_file, apply_patch, write_file, run_tests, lint_or_typecheck, run_bash, ask_user, plan_task, web_search, respond_only`.
- **Input per example**: the current user prompt + prior conversation/action history (user messages interleaved with the agent's past tool calls and their result summaries) + session metadata (user tier, workspace language mix, lines-of-code, git-dirty flag, open files, last CI status, remaining token budget, turn index, elapsed seconds).
- **Text**: short, informal, **code-mixed Korean (~64%) + English** tech terms/paths/identifiers.
- **Data**: SYNTHETIC simulated agent sessions, **70k** training examples, **class-imbalanced** (edit_file 16% … web_search 1.8%). **Metric = macro-F1** (unweighted mean of per-class F1).

## Hard constraints
- Code-submission competition: the final **model + inference code must be ≤1GB total**, run **OFFLINE** on a single **NVIDIA T4 (16GB)**, **inference ≤10 min**, pip install ≤10 min.
- The **TEST SET IS HIDDEN** — I never see it at train time, so classic **pseudo-labeling / transductive learning on the test set is essentially blocked**.
- Training GPU available locally: RTX 4070 Ti 12GB.

## What I've already tried (do NOT just repeat these)
- Fine-tuned **xlm-roberta-base → ~0.69–0.71 macro-F1** (my best single model). max_len 512 slightly beats 256 (+0.014).
- **mdeberta-v3-base** (fp32; bf16 NaNs due to disentangled-attention overflow) → **WORSE than XLM-R** on this task.
- **Per-class logit-bias post-processing** → +0.005–0.007.
- **Stacking structured metadata features** (last action, turn index, open-file count, CI status, history action counts) into a GBM on top of transformer probabilities → **did NOT help** (the transformer already captures these from the serialized history text).
- Classical TF-IDF on current-prompt-only caps ~0.68.
- **Confirmed noise ceiling**: ~40% of the data (the read/grep/glob/list "file-navigation" cluster) sits at **F1≈0.5** because the labels are genuinely **ambiguous / partly noise** — multiple actions are valid for the same prompt, and **99% of the model's errors stay within the correct semantic cluster** (it knows the *type* of action but not the exact one). This looks irreducible.
- My actual leaderboard score is **0.690**; the qualification cut is **~0.776 (top-12 of 868 teams)**. I need every bit.

## Research questions (answer each with evidence + source links)
1. **Winning-solution playbook**: What do winners of *similar* noisy/imbalanced multilingual short-text classification competitions (Kaggle, DACON) actually do to gain the last 3–8% of macro-F1? Link concrete writeups and quantify the reported gains.
2. **Better model under tight compute**: Are there 2024–2026 models/methods that *reliably* beat a fine-tuned xlm-roberta/DeBERTa encoder for short-text classification within ≤1GB & offline? Evaluate: **ModernBERT**, **LoRA-tuned small decoder LLMs** (e.g. Qwen2.5-0.5B/1.5B) as classifiers, **gte/e5 multilingual embeddings + classifier head**, and any Korean-specialized encoders. Give measured comparisons.
3. **Noise-robust training for macro-F1**: For confusable classes with irreducible label noise, which methods best maximize macro-F1 — **GCE / symmetric CE / bi-tempered logistic loss, co-teaching, label smoothing, soft/sigmoid-F1 surrogate loss, logit adjustment for class imbalance**? Rank by *reliable* gain and give tuning guidance.
4. **Train-time-only augmentation/regularization** (test is hidden): Rank by reliability — **R-Drop, FGM/AWP adversarial training, EDA, back-translation, mixup/manifold-mixup for text, consistency regularization**. Which are worth it for short code-mixed Korean text?
5. **Post-processing & ensembling**: Best practice for **per-class threshold / logit-bias** optimization to maximize macro-F1, and **ensembling** (multi-seed / multi-fold / multi-model / stacking) — what actually moves the needle and by how much.

## Required output format
- A **ranked table** of techniques by (expected macro-F1 gain × fit-to-my-constraints). Columns: technique · what it is · concrete expected gain · implementation effort · fits ≤1GB/offline/T4? · source link.
- **Explicitly call out anything I may have MISSED**, and any technique that could plausibly give a **step-change** (not just +0.005).
- An honest verdict: **is ~0.776 realistically reachable** given the noise ceiling, and **what is the single highest-leverage path**?
- Prefer primary sources (arXiv, official competition writeups, GitHub) over blogs. Cite everything.

---
*Context if asked: DACON "2026 AI·SW중심대학 디지털 경진대회 : AI부문", competition 236694.*

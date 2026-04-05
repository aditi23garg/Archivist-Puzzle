# 📚 The Archivist's Puzzle — HackRush 2026

> Reconstructing shuffled narrative pages using NLP and sentence embeddings.

---

## 🧩 Problem Statement

Given collections of text pages from mystery novels with their original ordering removed, reconstruct the correct sequence of pages using algorithmic or machine learning techniques.

Two test books were provided:
- **BookA** — 147 shuffled pages
- **BookB** — 57 shuffled pages

A training dataset (The Mysterious Affair at Styles, in correct order) was also provided as reference.

Submissions were evaluated using a **Kendall Tau-based score** normalized to [0, 1], where 0.5 = random ordering and 1.0 = perfect ordering.

---

## 💡 Core Idea

Pages that belong next to each other should feel similar. Specifically:

> The **ending of page N** should naturally flow into the **beginning of page N+1**.

They share the same characters, continue the same scene, and maintain the same narrative thread. We reduced the problem to one question:

**"For every pair of pages, how likely is page B to come right after page A?"**

---

## 🔧 Approach

### Step 1 — Text Segmentation
For each page, we extracted:
- **Tail** — last 200 words (what flows *out* of this page)
- **Head** — first 200 words (what flows *into* the next page)

### Step 2 — Sentence Embeddings
We used `all-MiniLM-L6-v2` from HuggingFace's `sentence-transformers` library to convert text segments into numerical vectors. This model understands the *meaning* of text rather than just matching words — so semantically similar sentences score high even without shared vocabulary.

### Step 3 — Transition Score Matrix
We built an **N×N matrix** where entry `[i][j]` = cosine similarity between the tail of page i and the head of page j.

```
transition[i][j] = cosine_similarity(tail_embedding[i], head_embedding[j])
```

Higher score = page j more likely follows page i.

### Step 4 — Greedy Chain Ordering
Starting from 50 candidate first pages, we greedily built an ordering by always choosing the highest-scoring unvisited next page. We kept the chain with the best total score.

### Step 5 — 2-opt Refinement
We applied **2-opt local search** (adapted from the Travelling Salesman Problem) to improve the ordering. This tries reversing every sub-segment of the current chain — if reversing pages 5 through 12 improves the total score, we keep that change. We repeat until no improvement is found.

---

## 🧪 Experiments

We ran 13 systematic experiments across the following approaches:

| # | Approach | Score |
|---|---|---|
| 1 | TF-IDF word matching | 0.5257 |
| **2** | **all-MiniLM-L6-v2, n=200 (best)** | **0.5470** |
| 3 | all-mpnet-base-v2 (large model) | 0.4700 |
| 4 | XGBoost trained on training book | 0.3990 |
| 5 | Beam search ordering | 0.4640 |
| 6 | All starting pages + 2-opt | 0.4597 |
| 7 | Cross-encoder rescoring | 0.5068 |
| 8 | Ensemble of 5 window sizes | 0.5415 |
| 9 | thenlper/gte-small | 0.4686 |
| 10 | Rich linguistic features + bonuses | 0.5417 |
| 11 | multi-qa model | 0.4730 |
| 12 | all-MiniLM-L6-v2, n=150 | 0.5127 |
| 13 | all-MiniLM-L6-v2, n=100 | not submitted |

---

## 🔑 Key Findings

**Simpler is better for this problem.**
The most complex models (gte-small, mpnet) consistently scored worse despite higher internal similarity scores. Powerful models make overconfident wrong connections that cascade through the entire ordering.

**Internal scores don't predict submission scores.**
gte-small scored 0.89 on our internal metric but only 0.47 on the leaderboard. The Kendall Tau metric rewards relative pairwise ordering, which behaves differently from raw similarity scores.

**The signal is fundamentally weak.**
Diagnostic analysis showed the true next page appears in the top-5 candidates only **28.2% of the time**. This means our greedy algorithm makes wrong choices 72% of the time — not because of a bad algorithm, but because consecutive literary pages are not always semantically similar at their boundaries. Authors shift scenes, change perspective, and jump in time even between adjacent pages.

**n=200 words is the sweet spot.**
Testing window sizes from 100 to 300 words showed that 200 words gave the best balance between specificity and context.

---

## 📊 Final Result

| Metric | Value |
|---|---|
| **Final Score** | **0.5470** |
| BookA pages | 147 |
| BookB pages | 57 |
| Evaluation | Kendall Tau normalized |
| Best model | all-MiniLM-L6-v2 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| sentence-transformers | Page embeddings |
| scikit-learn | Cosine similarity |
| pandas | Data handling |
| numpy | Matrix operations |
| xgboost | Pairwise classification (experimented) |

---

## 📁 Repository Structure

```
Archivist-Puzzle/
│
├── Codes/                                      # All solution scripts
│   ├── solve.py                                # v1  — TF-IDF baseline
│   ├── solve_v2.py                             # v2  — Best submission (0.5470) ⭐
│   ├── solve_v3.py                             # v3  — Large mpnet model
│   ├── solve_v4.py                             # v4  — XGBoost approach
│   ├── solve_v5.py                             # v5  — Beam search ordering
│   ├── solve_v6.py                             # v6  — All starts + 2-opt
│   ├── solve_v7.py                             # v7  — Cross-encoder rescoring
│   ├── solve_v8.py                             # v8  — Ensemble of 5 window sizes
│   ├── solve_v9.py                             # v9  — multi-qa model
│   ├── solve_v10.py                            # v10 — gte-small model
│   ├── solve_v11.py                            # v11 — Rich linguistic features
│   ├── solve_v12.py                            # v12 — n=150 window experiment
│   ├── solve_v13.py                            # v13 — n=100 window experiment
│   ├── solve_final.py                          # Final combined attempt
│   ├── test_models.py                          # Model comparison script
│   ├── diagnose.py                             # Window size diagnostic
│   ├── analyze.py                              # Top-5 accuracy analysis
│   ├── compare.py                              # Model vs ensemble comparison
│   ├── check.py                                # Data inspection script
│   ├── check_diff.py                           # Submission diff checker
│   └── check_v2.py                             # Submission v2 recreation
│
├── Files/                                      # All data and submission files
│   │
│   ├── # ── Input Data ──────────────────────────────
│   ├── BookA_test.csv                          # Shuffled pages — Book A (147 pages)
│   ├── BookB_test.csv                          # Shuffled pages — Book B (57 pages)
│   ├── Mysterious_Affair_at_Styles_Train_Data.csv  # Training data (correct order)
│   ├── sample_submission.csv                   # Submission format reference
│   │
│   ├── # ── Final Submissions ───────────────────────
│   ├── BookA.csv                               # Final submission — Book A ⭐
│   ├── BookB.csv                               # Final submission — Book B ⭐
│   │
│   ├── # ── All Submission Versions ─────────────────
│   ├── BookA_v6.csv                            # Book A — submission v6
│   ├── BookA_v7.csv                            # Book A — submission v7
│   ├── BookA_v8.csv                            # Book A — submission v8
│   ├── BookA_v9.csv                            # Book A — submission v9
│   ├── BookA_v10.csv                           # Book A — submission v10
│   ├── BookA_v11.csv                           # Book A — submission v11
│   ├── BookA_v12.csv                           # Book A — submission v12
│   ├── BookA_v13.csv                           # Book A — submission v13
│   ├── BookA_recreation.csv                    # Book A — v2 recreation attempt
│   ├── BookA_v2_recreation.csv                 # Book A — v2 recreation backup
│   ├── BookB_v6.csv                            # Book B — submission v6
│   ├── BookB_v7.csv                            # Book B — submission v7
│   ├── BookB_v8.csv                            # Book B — submission v8
│   ├── BookB_v9.csv                            # Book B — submission v9
│   ├── BookB_v10.csv                           # Book B — submission v10
│   ├── BookB_v11.csv                           # Book B — submission v11
│   ├── BookB_v12.csv                           # Book B — submission v12
│   ├── BookB_v13.csv                           # Book B — submission v13
│   ├── BookB_recreation.csv                    # Book B — v2 recreation attempt
│   └── BookB_v2_recreation.csv                 # Book B — v2 recreation backup
│
├── LICENSE                                     # MIT License
└── README.md                                   # This file
```

---

## 🚀 How To Run

### Install dependencies
```bash
pip install pandas numpy scikit-learn sentence-transformers xgboost
```

### Clone the repository
```bash
git clone https://github.com/aditi23garg/Archivist-Puzzle.git
cd Archivist-Puzzle
```

### Copy data files to working directory
```bash
cp Files/*.csv .
```

### Run best solution
```bash
python Codes/solve_v2.py
```

This will generate `BookA.csv` and `BookB.csv` ready for submission.

---

## 🔮 Future Improvements

- **Better top-5 accuracy** — the core bottleneck is that correct next pages are hard to identify. Better narrative-specific embeddings could help.
- **LLM pairwise ranking** — use a large language model to directly judge "does page B follow page A?" for top candidate pairs.
- **Chapter detection** — identify chapter boundaries first, then order within chapters.
- **Dialogue tracking** — follow speaker patterns and open/close quotes across pages.
- **Named entity chains** — track character appearances more precisely across the full book.

---

**HackRush 2026 — ML Problem Statement 3**
*The Archivist's Puzzle: Reconstructing a Lost Narrative*

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
├── BookA_test.csv                          # Shuffled pages for Book A
├── BookB_test.csv                          # Shuffled pages for Book B
├── Mysterious_Affair_at_Styles_Train_Data.csv  # Training data (correct order)
├── sample_submission.csv                   # Submission format reference
├── solve.py                                # v1 — TF-IDF baseline
├── solve_v2.py                             # v2 — Best submission (0.5470)
├── solve_v4.py                             # v4 — XGBoost approach
├── solve_v5.py                             # v5 — Beam search
├── solve_v8.py                             # v8 — Ensemble approach
├── solve_v11.py                            # v11 — Rich linguistic features
├── BookA.csv                               # Final submission for Book A
├── BookB.csv                               # Final submission for Book B
└── README.md                               # This file
```

---

## 🚀 How To Run

### Install dependencies
```bash
pip install pandas numpy scikit-learn sentence-transformers
```

### Run best solution
```bash
python solve_v2.py
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

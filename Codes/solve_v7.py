import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("Loading data...")
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')
print(f"BookA: {len(bookA_df)} | BookB: {len(bookB_df)}")

# ── LOAD BOTH MODELS ───────────────────────────────────
# Model 1: Bi-encoder (our best so far)
print("Loading bi-encoder...")
bi_model = SentenceTransformer('all-MiniLM-L6-v2')

# Model 2: Cross-encoder (new, smarter)
print("Loading cross-encoder...")
cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Both models ready!")

# ── TEXT HELPERS ───────────────────────────────────────
def get_tail(text, n=150):
    words = str(text).split()
    return " ".join(words[-n:])

def get_head(text, n=150):
    words = str(text).split()
    return " ".join(words[:n])

# ── BUILD TRANSITION MATRIX ────────────────────────────
def build_transition_matrix(df):
    texts = df['text'].fillna("").tolist()
    n = len(texts)
    tails = [get_tail(t) for t in texts]
    heads = [get_head(t) for t in texts]

    # ── Signal 1: Bi-encoder similarity (fast) ──
    print("  Bi-encoder embeddings...")
    tail_embs = bi_model.encode(tails, show_progress_bar=True)
    head_embs = bi_model.encode(heads, show_progress_bar=True)
    sim_bi = cosine_similarity(tail_embs, head_embs)

    # ── Signal 2: Cross-encoder scores (smart) ──
    # Cross-encoder is slow so we only run it on TOP candidates
    # For each page i, find top 10 candidate next pages using bi-encoder
    # Then re-score those 10 with cross-encoder
    print("  Cross-encoder rescoring top candidates...")
    sim_cross = np.zeros((n, n))

    for i in range(n):
        if i % 20 == 0:
            print(f"    Page {i}/{n}...")

        # Get top 15 candidates from bi-encoder for this page
        bi_scores = sim_bi[i].copy()
        bi_scores[i] = -999  # exclude self
        top_candidates = np.argsort(bi_scores)[::-1][:15]

        # Build pairs: (tail of page i, head of candidate j)
        pairs = [(tails[i], heads[j]) for j in top_candidates]

        # Cross-encoder scores all pairs at once
        ce_scores = cross_model.predict(pairs)

        # Fill in cross-encoder scores for top candidates
        for idx, j in enumerate(top_candidates):
            sim_cross[i, j] = ce_scores[idx]

    # Normalize cross-encoder scores to 0-1 range
    min_ce = sim_cross[sim_cross > 0].min() if (sim_cross > 0).any() else 0
    max_ce = sim_cross.max()
    if max_ce > min_ce:
        sim_cross = np.where(
            sim_cross > 0,
            (sim_cross - min_ce) / (max_ce - min_ce),
            0
        )

    # ── Combine both signals ──
    # For top candidates: use cross-encoder (smarter)
    # For others: use bi-encoder (as fallback)
    combined = np.where(
        sim_cross > 0,
        0.4 * sim_bi + 0.6 * sim_cross,  # cross-encoder available
        sim_bi                             # fallback to bi-encoder
    )
    np.fill_diagonal(combined, -999)
    return combined

# ── ORDERING ALGORITHMS ────────────────────────────────
def greedy_chain(trans, start):
    n = trans.shape[0]
    visited = [False] * n
    order = [start]
    visited[start] = True
    total = 0.0
    current = start
    for _ in range(n - 1):
        scores = trans[current].copy()
        for vi, v in enumerate(visited):
            if v:
                scores[vi] = -999
        nxt = int(np.argmax(scores))
        total += trans[current, nxt]
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return order, total

def path_score(order, trans):
    return sum(trans[order[k], order[k+1]]
               for k in range(len(order)-1))

def two_opt(order, trans, max_iters=500):
    order = list(order)
    n = len(order)
    best_score = path_score(order, trans)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                new_order = (order[:i+1]
                           + order[i+1:j+1][::-1]
                           + order[j+1:])
                new_score = path_score(new_order, trans)
                if new_score > best_score + 1e-9:
                    order = new_order
                    best_score = new_score
                    improved = True
    return order, best_score

# ── PROCESS EACH BOOK ──────────────────────────────────
def process_book(df, name):
    print(f"\n{'='*40}")
    print(f"Processing {name} ({len(df)} pages)")
    print(f"{'='*40}")
    shuffled_pages = df['page'].tolist()

    trans = build_transition_matrix(df)

    n = len(df)
    best_order, best_score = None, -np.inf

    print(f"  Trying all {n} starting pages...")
    for s in range(n):
        o, sc = greedy_chain(trans, s)
        if sc > best_score:
            best_score, best_order = sc, o
    print(f"  Score after greedy: {best_score:.4f}")

    best_order, best_score = two_opt(best_order, trans)
    print(f"  Score after 2-opt:  {best_score:.4f}")

    rows = [{'original_page': i+1,
             'shuffled_page': shuffled_pages[idx]}
            for i, idx in enumerate(best_order)]
    result = pd.DataFrame(rows)

    assert result['shuffled_page'].nunique() == len(df)
    assert list(result['original_page']) == list(range(1, len(df)+1))
    print("  Validation passed!")

    result.to_csv(f'{name}.csv', index=False)
    print(f"  Saved {name}.csv")
    print(result.head(10).to_string(index=False))

process_book(bookA_df, 'BookA')
process_book(bookB_df, 'BookB')
print("\nDone! Submit BookA.csv and BookB.csv")
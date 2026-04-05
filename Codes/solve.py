import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# ── STEP 1: Load the data ──────────────────────────────
print("Loading CSV files...")
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')
print(f"BookA: {len(bookA_df)} pages")
print(f"BookB: {len(bookB_df)} pages")

# ── STEP 2: Helper functions ───────────────────────────
def get_tail(text, n=150):
    # Get the LAST 150 words of a page
    # (this is what flows INTO the next page)
    words = str(text).split()
    return " ".join(words[-n:])

def get_head(text, n=150):
    # Get the FIRST 150 words of a page
    # (this receives flow from the previous page)
    words = str(text).split()
    return " ".join(words[:n])

# ── STEP 3: Build transition matrix ───────────────────
# transition[i][j] = score of "page j comes right after page i"
def build_transition_matrix(df):
    texts = df['text'].fillna("").tolist()
    n = len(texts)

    tails = [get_tail(t) for t in texts]
    heads = [get_head(t) for t in texts]

    print("  Converting text to TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # use single words AND pairs of words
        min_df=1,
        max_features=20000,
        sublinear_tf=True     # smooth the word counts with log
    )
    # Fit on all text so vocabulary is shared
    vectorizer.fit(tails + heads)

    tail_vecs = vectorizer.transform(tails)
    head_vecs = vectorizer.transform(heads)

    print("  Computing similarity scores...")
    # For each pair (i, j): how similar is end of i to start of j?
    sim = cosine_similarity(tail_vecs, head_vecs)

    # A page cannot follow itself
    np.fill_diagonal(sim, -999)

    return sim

# ── STEP 4: Greedy chain ───────────────────────────────
# Start at a page, always go to the best unvisited next page
def greedy_chain(trans, start):
    n = trans.shape[0]
    visited = [False] * n
    order = [start]
    visited[start] = True
    total_score = 0.0
    current = start

    for _ in range(n - 1):
        scores = trans[current].copy()
        for vi, v in enumerate(visited):
            if v:
                scores[vi] = -999
        next_page = int(np.argmax(scores))
        total_score += trans[current, next_page]
        order.append(next_page)
        visited[next_page] = True
        current = next_page

    return order, total_score

# ── STEP 5: 2-opt improvement ─────────────────────────
# Try reversing segments of the order to improve the score
def two_opt(order, trans, max_iters=300):
    order = list(order)
    n = len(order)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                old = trans[order[i], order[i+1]]
                if j + 1 < n:
                    old += trans[order[j], order[j+1]]
                new = trans[order[i], order[j]]
                if j + 1 < n:
                    new += trans[order[i+1], order[j+1]]
                if new > old + 1e-9:
                    order[i+1:j+1] = order[i+1:j+1][::-1]
                    improved = True

    return order

def path_score(order, trans):
    return sum(trans[order[k], order[k+1]] for k in range(len(order) - 1))

# ── STEP 6: Full pipeline for one book ────────────────
def process_book(df, name):
    print(f"\n{'='*40}")
    print(f"Processing {name} ({len(df)} pages)")
    print(f"{'='*40}")

    shuffled_pages = df['page'].tolist()

    # Build the n x n score matrix
    trans = build_transition_matrix(df)

    # Try 50 different starting pages, keep best
    n = len(df)
    avg_out = trans.mean(axis=1)
    starts = np.argsort(avg_out)[::-1][:min(50, n)]

    best_order, best_score = None, -np.inf
    print(f"  Trying {len(starts)} starting pages for greedy chain...")
    for s in starts:
        o, sc = greedy_chain(trans, int(s))
        if sc > best_score:
            best_score, best_order = sc, o
    print(f"  Score after greedy:  {best_score:.4f}")

    # Improve with 2-opt
    best_order = two_opt(best_order, trans)
    print(f"  Score after 2-opt:   {path_score(best_order, trans):.4f}")

    # Build output dataframe
    rows = []
    for position, df_index in enumerate(best_order):
        rows.append({
            'original_page': position + 1,
            'shuffled_page': shuffled_pages[df_index]
        })
    result = pd.DataFrame(rows)

    # Validate
    assert result['shuffled_page'].nunique() == len(df), "ERROR: duplicates found!"
    assert list(result['original_page']) == list(range(1, len(df)+1)), "ERROR: not 1-based!"
    print(f"  Validation passed!")

    result.to_csv(f'{name}.csv', index=False)
    print(f"  Saved {name}.csv")
    print(result.head(10).to_string(index=False))

# ── RUN ───────────────────────────────────────────────
process_book(bookA_df, 'BookA')
process_book(bookB_df, 'BookB')
print("\nDone! BookA.csv and BookB.csv are ready to submit.")
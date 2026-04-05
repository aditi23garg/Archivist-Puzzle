import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading CSV files...")
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')
print(f"BookA: {len(bookA_df)} pages")
print(f"BookB: {len(bookB_df)} pages")

print("Loading embedding model...")
model = SentenceTransformer('all-mpnet-base-v2')
print("Model ready!")

def get_tail(text, n=250):
    words = str(text).split()
    return " ".join(words[-n:])

def get_head(text, n=250):
    words = str(text).split()
    return " ".join(words[:n])

def build_transition_matrix(df):
    texts = df['text'].fillna("").tolist()
    n = len(texts)
    tails = [get_tail(t) for t in texts]
    heads = [get_head(t) for t in texts]

    print("  Embedding tails...")
    tail_embs = model.encode(tails, show_progress_bar=True)
    print("  Embedding heads...")
    head_embs = model.encode(heads, show_progress_bar=True)

    # Also embed full pages for global context
    print("  Embedding full pages...")
    full_embs = model.encode(
        [" ".join(str(t).split()[:400]) for t in texts],
        show_progress_bar=True
    )

    print("  Computing similarity scores...")
    sim_tail_head = cosine_similarity(tail_embs, head_embs)
    sim_full_full = cosine_similarity(full_embs, full_embs)

    # Combine: 70% tail->head flow + 30% global context
    combined = 0.7 * sim_tail_head + 0.3 * sim_full_full
    np.fill_diagonal(combined, -999)
    return combined

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
    return sum(trans[order[k], order[k+1]] for k in range(len(order)-1))

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
                new_order = order[:i+1] + order[i+1:j+1][::-1] + order[j+1:]
                new_score = path_score(new_order, trans)
                if new_score > best_score + 1e-9:
                    order = new_order
                    best_score = new_score
                    improved = True
    return order, best_score

def process_book(df, name):
    print(f"\n{'='*40}")
    print(f"Processing {name} ({len(df)} pages)")
    print(f"{'='*40}")
    shuffled_pages = df['page'].tolist()

    trans = build_transition_matrix(df)

    n = len(df)
    # Try ALL pages as starting points for smaller book
    # For larger book, try top 80
    n_starts = n if n <= 57 else 80
    avg_out = trans.mean(axis=1)
    starts = np.argsort(avg_out)[::-1][:n_starts]

    best_order, best_score = None, -np.inf
    print(f"  Trying {len(starts)} starting pages...")
    for s in starts:
        o, sc = greedy_chain(trans, int(s))
        if sc > best_score:
            best_score, best_order = sc, o
    print(f"  Score after greedy: {best_score:.4f}")

    best_order, best_score = two_opt(best_order, trans)
    print(f"  Score after 2-opt:  {best_score:.4f}")

    rows = [{'original_page': i+1, 'shuffled_page': shuffled_pages[idx]}
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
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading data...")
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready!")

def get_tail(text, n):
    words = str(text).split()
    return " ".join(words[-n:])

def get_head(text, n):
    words = str(text).split()
    return " ".join(words[:n])

def build_matrix(df, n_words):
    texts = df['text'].fillna("").tolist()
    tails = [get_tail(t, n_words) for t in texts]
    heads = [get_head(t, n_words) for t in texts]
    tail_embs = model.encode(tails, show_progress_bar=False)
    head_embs = model.encode(heads, show_progress_bar=False)
    sim = cosine_similarity(tail_embs, head_embs)
    np.fill_diagonal(sim, -999)
    return sim

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

def best_greedy(trans):
    n = trans.shape[0]
    best_order, best_score = None, -np.inf
    for s in range(n):
        o, sc = greedy_chain(trans, s)
        if sc > best_score:
            best_score, best_order = sc, o
    return best_order, best_score

def process_book(df, name):
    print(f"\n{'='*40}")
    print(f"Processing {name} ({len(df)} pages)")
    print(f"{'='*40}")
    shuffled_pages = df['page'].tolist()
    n = len(df)

    # Try multiple window sizes and ensemble them
    all_matrices = []
    for n_words in [100, 150, 200, 250, 300]:
        print(f"  Building matrix n_words={n_words}...")
        m = build_matrix(df, n_words)
        all_matrices.append(m)

    # Strategy 1: Use each matrix individually, pick best ordering
    print("  Testing individual matrices...")
    best_order_individual = None
    best_score_individual = -np.inf
    for i, m in enumerate(all_matrices):
        o, sc = best_greedy(m)
        o, sc = two_opt(o, m)
        # Evaluate this ordering on ALL matrices combined
        combined = sum(all_matrices) / len(all_matrices)
        eval_score = path_score(o, combined)
        print(f"    n_words={[100,150,200,250,300][i]} → combined eval score: {eval_score:.4f}")
        if eval_score > best_score_individual:
            best_score_individual = eval_score
            best_order_individual = o

    # Strategy 2: Average all matrices (ensemble)
    print("  Testing ensemble matrix...")
    combined = sum(all_matrices) / len(all_matrices)
    o_ens, sc_ens = best_greedy(combined)
    o_ens, sc_ens = two_opt(o_ens, combined)
    print(f"    Ensemble score: {sc_ens:.4f}")

    # Pick whichever is better
    if sc_ens >= best_score_individual:
        best_order = o_ens
        print(f"  → Using ensemble ordering (score: {sc_ens:.4f})")
    else:
        best_order = best_order_individual
        print(f"  → Using individual best ordering (score: {best_score_individual:.4f})")

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
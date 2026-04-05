import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Recreate EXACT submission 2 conditions
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_tail(text, n=200):
    return " ".join(str(text).split()[-n:])

def get_head(text, n=200):
    return " ".join(str(text).split()[:n])

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
            if v: scores[vi] = -999
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
                new_order = (order[:i+1] +
                            order[i+1:j+1][::-1] +
                            order[j+1:])
                new_score = path_score(new_order, trans)
                if new_score > best_score + 1e-9:
                    order = new_order
                    best_score = new_score
                    improved = True
    return order, best_score

for book_name, df in [('BookA', bookA_df), ('BookB', bookB_df)]:
    texts = df['text'].fillna("").tolist()
    tails = [get_tail(t) for t in texts]
    heads = [get_head(t) for t in texts]
    te = model.encode(tails, show_progress_bar=False)
    he = model.encode(heads, show_progress_bar=False)
    sim = cosine_similarity(te, he)
    np.fill_diagonal(sim, -999)

    # Submission 2 used top 50 starts
    avg_out = sim.mean(axis=1)
    starts = np.argsort(avg_out)[::-1][:50]

    best_order, best_score = None, -np.inf
    for s in starts:
        o, sc = greedy_chain(sim, int(s))
        if sc > best_score:
            best_score, best_order = sc, o

    best_order, best_score = two_opt(best_order, sim)

    shuffled = df['page'].tolist()
    rows = [{'original_page': i+1, 'shuffled_page': shuffled[idx]}
            for i, idx in enumerate(best_order)]
    result = pd.DataFrame(rows)
    result.to_csv(f'{book_name}_recreation.csv', index=False)
    print(f"{book_name}: score={best_score:.4f}")
    print(result.head(10).to_string(index=False))
    print()
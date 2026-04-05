import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')

# Load BOTH models
print("Loading models...")
model1 = SentenceTransformer('all-MiniLM-L6-v2')
model2 = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def get_tail(text, n=200):
    return " ".join(str(text).split()[-n:])

def get_head(text, n=200):
    return " ".join(str(text).split()[:n])

def get_matrix(df, model, n=200):
    texts = df['text'].fillna("").tolist()
    tails = [get_tail(t, n) for t in texts]
    heads = [get_head(t, n) for t in texts]
    te = model.encode(tails, show_progress_bar=False)
    he = model.encode(heads, show_progress_bar=False)
    sim = cosine_similarity(te, he)
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
            if v: scores[vi] = -999
        nxt = int(np.argmax(scores))
        total += trans[current, nxt]
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return order, total

def path_score(order, trans):
    return sum(trans[order[k], order[k+1]] for k in range(len(order)-1))

def best_order(trans):
    n = trans.shape[0]
    best_o, best_s = None, -np.inf
    for s in range(n):
        o, sc = greedy_chain(trans, s)
        if sc > best_s:
            best_s, best_o = sc, o
    return best_o, best_s

print("\nComputing matrices for BookA...")
m1_A = get_matrix(bookA_df, model1, 200)
m2_A = get_matrix(bookA_df, model2, 200)
m_avg_A = (m1_A + m2_A) / 2

print("Computing matrices for BookB...")
m1_B = get_matrix(bookB_df, model1, 200)
m2_B = get_matrix(bookB_df, model2, 200)
m_avg_B = (m1_B + m2_B) / 2

print("\n=== BookA orderings evaluated on AVERAGED matrix ===")
for label, m_build, m_eval in [
    ("model1 only",   m1_A,    m_avg_A),
    ("model2 only",   m2_A,    m_avg_A),
    ("averaged",      m_avg_A, m_avg_A),
]:
    o, _ = best_order(m_build)
    sc = path_score(o, m_eval)
    print(f"  {label:20s} → eval score: {sc:.4f}")

print("\n=== BookB orderings evaluated on AVERAGED matrix ===")
for label, m_build, m_eval in [
    ("model1 only",   m1_B,    m_avg_B),
    ("model2 only",   m2_B,    m_avg_B),
    ("averaged",      m_avg_B, m_avg_B),
]:
    o, _ = best_order(m_build)
    sc = path_score(o, m_eval)
    print(f"  {label:20s} → eval score: {sc:.4f}")
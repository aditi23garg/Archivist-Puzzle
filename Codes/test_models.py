import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')

MODELS = [
    'all-MiniLM-L6-v2',
    'BAAI/bge-small-en-v1.5',
    'BAAI/bge-base-en-v1.5',
    'thenlper/gte-small',
    'multi-qa-mpnet-base-dot-v1',
]

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
    return sum(trans[order[k], order[k+1]]
               for k in range(len(order)-1))

def best_greedy(trans):
    n = trans.shape[0]
    best_o, best_s = None, -np.inf
    for s in range(n):
        o, sc = greedy_chain(trans, s)
        if sc > best_s:
            best_s, best_o = sc, o
    return best_o, best_s

# Store all matrices for ensemble testing
all_matrices_A = {}
all_matrices_B = {}
individual_scores = []

for model_name in MODELS:
    print(f"\n{'='*45}")
    print(f"Testing: {model_name}")
    try:
        model = SentenceTransformer(model_name)

        mA = get_matrix(bookA_df, model)
        mB = get_matrix(bookB_df, model)

        all_matrices_A[model_name] = mA
        all_matrices_B[model_name] = mB

        oA, sA = best_greedy(mA)
        oB, sB = best_greedy(mB)

        norm_A = sA / len(bookA_df)
        norm_B = sB / len(bookB_df)
        combined = (norm_A + norm_B) / 2

        print(f"  BookA normalized: {norm_A:.4f}")
        print(f"  BookB normalized: {norm_B:.4f}")
        print(f"  Combined:         {combined:.4f}")

        individual_scores.append({
            'model': model_name,
            'mA': mA,
            'mB': mB,
            'norm_A': norm_A,
            'norm_B': norm_B,
            'combined': combined
        })

    except Exception as e:
        print(f"  ERROR: {e}")

# Test all ensemble combinations
print(f"\n{'='*45}")
print("TESTING ENSEMBLE COMBINATIONS")
print(f"{'='*45}")

model_names = list(all_matrices_A.keys())
n_models = len(model_names)

best_ensemble_score = -np.inf
best_ensemble_combo = None

# Test every possible combination of 2, 3, 4, 5 models
from itertools import combinations
for r in range(2, n_models + 1):
    for combo in combinations(model_names, r):
        # Average matrices
        avg_A = sum(all_matrices_A[m] for m in combo) / len(combo)
        avg_B = sum(all_matrices_B[m] for m in combo) / len(combo)

        oA, sA = best_greedy(avg_A)
        oB, sB = best_greedy(avg_B)

        norm_A = sA / len(bookA_df)
        norm_B = sB / len(bookB_df)
        combined = (norm_A + norm_B) / 2

        if combined > best_ensemble_score:
            best_ensemble_score = combined
            best_ensemble_combo = combo

        print(f"  {' + '.join([m.split('/')[-1] for m in combo])}")
        print(f"    Combined: {combined:.4f}")

print(f"\n{'='*45}")
print("FINAL RANKING — INDIVIDUAL MODELS")
print(f"{'='*45}")
individual_scores.sort(key=lambda x: x['combined'], reverse=True)
for i, r in enumerate(individual_scores):
    print(f"  #{i+1} {r['model']:45s} → {r['combined']:.4f}")

print(f"\n{'='*45}")
print("BEST ENSEMBLE COMBINATION")
print(f"{'='*45}")
print(f"  Models: {best_ensemble_combo}")
print(f"  Score:  {best_ensemble_score:.4f}")
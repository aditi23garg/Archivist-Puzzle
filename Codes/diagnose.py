import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_tail(text, n=200):
    words = str(text).split()
    return " ".join(words[-n:])

def get_head(text, n=200):
    words = str(text).split()
    return " ".join(words[:n])

# Test different window sizes and see which gives highest matrix confidence
for n_words in [100, 150, 200, 250, 300]:
    texts = bookA_df['text'].fillna("").tolist()
    tails = [get_tail(t, n_words) for t in texts]
    heads = [get_head(t, n_words) for t in texts]
    
    tail_embs = model.encode(tails, show_progress_bar=False)
    head_embs = model.encode(heads, show_progress_bar=False)
    
    sim = cosine_similarity(tail_embs, head_embs)
    np.fill_diagonal(sim, -1)
    
    # How "confident" is our matrix?
    # High confidence = big gap between best and second best score for each row
    gaps = []
    for i in range(len(texts)):
        row = sim[i]
        sorted_row = np.sort(row)[::-1]
        gap = sorted_row[0] - sorted_row[1]
        gaps.append(gap)
    
    avg_gap = np.mean(gaps)
    avg_best = np.mean([np.max(sim[i]) for i in range(len(texts))])
    print(f"n_words={n_words:3d} | avg_best_score={avg_best:.4f} | avg_gap={avg_gap:.4f}")
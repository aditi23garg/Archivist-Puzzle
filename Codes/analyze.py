import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')
train_df = pd.read_csv('Mysterious_Affair_at_Styles_Train_Data.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_tail(text, n=200):
    return " ".join(str(text).split()[-n:])

def get_head(text, n=200):
    return " ".join(str(text).split()[:n])

# Key question: in the TRAINING data (correct order),
# what is the average similarity between consecutive pages?
# vs non-consecutive pages?
# This tells us if our similarity signal is meaningful at all.

print("Analyzing training data...")
texts = train_df['text'].fillna("").tolist()
tails = [get_tail(t) for t in texts]
heads = [get_head(t) for t in texts]

te = model.encode(tails, show_progress_bar=False)
he = model.encode(heads, show_progress_bar=False)
sim = cosine_similarity(te, he)

# Consecutive pairs (correct transitions)
consec_scores = []
for i in range(len(texts)-1):
    consec_scores.append(sim[i, i+1])

# Random non-consecutive pairs
random_scores = []
np.random.seed(42)
for _ in range(500):
    i = np.random.randint(0, len(texts))
    j = np.random.randint(0, len(texts))
    if abs(i-j) > 1:
        random_scores.append(sim[i, j])

print(f"Consecutive page similarity:     mean={np.mean(consec_scores):.4f}  std={np.std(consec_scores):.4f}")
print(f"Non-consecutive page similarity: mean={np.mean(random_scores):.4f}  std={np.std(random_scores):.4f}")
print(f"Separation (gap):                {np.mean(consec_scores) - np.mean(random_scores):.4f}")
print()
print("If gap is small → our similarity signal is weak")
print("If gap is large → our similarity signal is strong")
print()

# Also check: what % of consecutive pairs rank in top 5 most similar?
top5_correct = 0
for i in range(len(texts)-1):
    row = sim[i].copy()
    row[i] = -999
    top5 = np.argsort(row)[::-1][:5]
    if (i+1) in top5:
        top5_correct += 1

pct = top5_correct / (len(texts)-1) * 100
print(f"Consecutive page appears in top-5 candidates: {top5_correct}/{len(texts)-1} = {pct:.1f}%")
print()
print("If this % is high → greedy has a good chance of being right")
print("If this % is low  → greedy will make many wrong choices")
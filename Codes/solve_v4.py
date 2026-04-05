import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
import re
from collections import Counter

# ── LOAD DATA ─────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv('Mysterious_Affair_at_Styles_Train_Data.csv')
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')
print(f"Train: {len(train_df)} | BookA: {len(bookA_df)} | BookB: {len(bookB_df)}")

# ── EMBEDDING MODEL ───────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model ready!")

def get_tail(text, n=150):
    words = str(text).split()
    return " ".join(words[-n:])

def get_head(text, n=150):
    words = str(text).split()
    return " ".join(words[:n])

# ── FEATURE EXTRACTION ────────────────────────────────
# For a pair (page_i, page_j), extract features that tell us
# "how likely is page_j to come right after page_i?"

def extract_pair_features(tail_embs, head_embs, tail_texts, head_texts, i, j):
    """
    Extract a feature vector for the pair (i -> j)
    """
    features = []

    # Feature 1: Cosine similarity between tail of i and head of j
    cos_sim = cosine_similarity(
        tail_embs[i].reshape(1, -1),
        head_embs[j].reshape(1, -1)
    )[0][0]
    features.append(cos_sim)

    # Feature 2: Cosine similarity between head of i and head of j
    cos_sim2 = cosine_similarity(
        head_embs[i].reshape(1, -1),
        head_embs[j].reshape(1, -1)
    )[0][0]
    features.append(cos_sim2)

    # Feature 3: Shared word count between tail_i and head_j
    tail_words = set(re.findall(r'[a-z]+', tail_texts[i].lower()))
    head_words = set(re.findall(r'[a-z]+', head_texts[j].lower()))
    shared = len(tail_words & head_words)
    total = len(tail_words | head_words)
    features.append(shared / total if total > 0 else 0)

    # Feature 4: Shared capitalized words (names/places)
    tail_caps = set(w for w in tail_texts[i].split()
                   if len(w) > 2 and w[0].isupper())
    head_caps = set(w for w in head_texts[j].split()
                   if len(w) > 2 and w[0].isupper())
    shared_caps = len(tail_caps & head_caps)
    features.append(shared_caps)

    # Feature 5: Does tail of i end mid-sentence? (continuation signal)
    tail_stripped = tail_texts[i].strip()
    ends_with_punct = 1 if tail_stripped and tail_stripped[-1] in '.!?"' else 0
    features.append(ends_with_punct)

    # Feature 6: Length difference between pages
    len_i = len(str(tail_texts[i]).split())
    len_j = len(str(head_texts[j]).split())
    features.append(abs(len_i - len_j))

    return features

# ── BUILD TRAINING DATA FROM TRAIN BOOK ───────────────
print("\nBuilding training data from Styles book...")
train_texts = train_df['text'].fillna("").tolist()
n_train = len(train_texts)

train_tails = [get_tail(t) for t in train_texts]
train_heads = [get_head(t) for t in train_texts]

print("  Embedding training pages...")
train_tail_embs = model.encode(train_tails, show_progress_bar=True)
train_head_embs = model.encode(train_heads, show_progress_bar=True)

X = []  # feature vectors
y = []  # labels: 1 = consecutive, 0 = not consecutive

print("  Creating positive and negative pairs...")
# Positive pairs: consecutive pages (label = 1)
for i in range(n_train - 1):
    feat = extract_pair_features(
        train_tail_embs, train_head_embs,
        train_tails, train_heads, i, i+1
    )
    X.append(feat)
    y.append(1)

# Negative pairs: random non-consecutive pages (label = 0)
# We sample 3x as many negative pairs to balance the data
np.random.seed(42)
neg_count = 0
target_neg = (n_train - 1) * 3
while neg_count < target_neg:
    i = np.random.randint(0, n_train)
    j = np.random.randint(0, n_train)
    if abs(i - j) > 1:  # not consecutive
        feat = extract_pair_features(
            train_tail_embs, train_head_embs,
            train_tails, train_heads, i, j
        )
        X.append(feat)
        y.append(0)
        neg_count += 1

X = np.array(X)
y = np.array(y)
print(f"  Training pairs: {len(y)} ({sum(y)} positive, {len(y)-sum(y)} negative)")

# ── TRAIN XGBOOST ─────────────────────────────────────
print("\nTraining XGBoost...")
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
clf.fit(X, y)
print("XGBoost trained!")

# ── APPLY TO TEST BOOKS ───────────────────────────────
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

def build_xgb_transition_matrix(df):
    texts = df['text'].fillna("").tolist()
    n = len(texts)
    tails = [get_tail(t) for t in texts]
    heads = [get_head(t) for t in texts]

    print("  Embedding pages...")
    tail_embs = model.encode(tails, show_progress_bar=True)
    head_embs = model.encode(heads, show_progress_bar=True)

    print("  Computing XGBoost transition scores...")
    trans = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                feat = extract_pair_features(
                    tail_embs, head_embs, tails, heads, i, j
                )
                # Use probability of being a "consecutive" pair
                trans[i, j] = clf.predict_proba([feat])[0][1]
        if i % 20 == 0:
            print(f"    Row {i}/{n} done...")

    np.fill_diagonal(trans, -999)
    return trans

def process_book(df, name):
    print(f"\n{'='*40}")
    print(f"Processing {name} ({len(df)} pages)")
    print(f"{'='*40}")
    shuffled_pages = df['page'].tolist()

    trans = build_xgb_transition_matrix(df)

    n = len(df)
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
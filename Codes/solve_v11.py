import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

print("Loading data...")
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready!")

# ── FEATURE HELPERS ───────────────────────────────────

def get_segment(text, mode, n=200):
    words = str(text).split()
    if mode == 'tail':
        return " ".join(words[-n:])
    elif mode == 'head':
        return " ".join(words[:n])
    elif mode == 'mid':
        m = len(words) // 2
        return " ".join(words[max(0,m-n//2):m+n//2])

def get_names(text):
    """Extract proper nouns (likely character names)"""
    words = str(text).split()
    names = set()
    for i, w in enumerate(words):
        clean = re.sub(r'[^A-Za-z]', '', w)
        if len(clean) > 2 and clean[0].isupper() and i > 0:
            names.add(clean)
    return names

def get_rare_words(text, word_freq, max_freq=5):
    """Words that appear rarely across all pages — more informative"""
    words = re.findall(r'[a-z]+', str(text).lower())
    return set(w for w in words if word_freq.get(w, 0) <= max_freq and len(w) > 4)

def ends_incomplete(text):
    """Does the page end mid-sentence? Strong signal of continuation"""
    t = str(text).strip()
    if not t:
        return 0
    return 0 if t[-1] in '.!?"\'»' else 1

def starts_lowercase(text):
    """Does page start with lowercase? Suggests continuation from previous"""
    t = str(text).strip()
    if not t:
        return 0
    first_char = t[0]
    return 1 if first_char.islower() else 0

def dialogue_continuity(tail, head):
    """Is there open dialogue at end of tail continued in head?"""
    tail_quotes = tail.count('"')
    if tail_quotes % 2 == 1:  # odd = unclosed quote
        return 1
    return 0

# ── BUILD RICH TRANSITION MATRIX ─────────────────────

def build_rich_matrix(df):
    texts = df['text'].fillna("").tolist()
    n = len(texts)

    # Precompute word frequencies across all pages
    all_words = []
    for t in texts:
        all_words.extend(re.findall(r'[a-z]+', t.lower()))
    word_freq = Counter(all_words)

    # Precompute segments
    tails     = [get_segment(t, 'tail', 200) for t in texts]
    heads     = [get_segment(t, 'head', 200) for t in texts]
    mids      = [get_segment(t, 'mid',  150) for t in texts]
    tails_sm  = [get_segment(t, 'tail', 100) for t in texts]
    heads_sm  = [get_segment(t, 'head', 100) for t in texts]

    # Precompute linguistic features
    names      = [get_names(t) for t in texts]
    rare_words = [get_rare_words(t, word_freq) for t in texts]
    incomplete = [ends_incomplete(t) for t in texts]
    starts_lc  = [starts_lowercase(t) for t in texts]

    # Embed all segments
    print("  Embedding tails (200)...")
    te200 = model.encode(tails,    show_progress_bar=True)
    print("  Embedding heads (200)...")
    he200 = model.encode(heads,    show_progress_bar=True)
    print("  Embedding tails (100)...")
    te100 = model.encode(tails_sm, show_progress_bar=True)
    print("  Embedding heads (100)...")
    he100 = model.encode(heads_sm, show_progress_bar=True)
    print("  Embedding mids...")
    me150 = model.encode(mids,     show_progress_bar=True)

    # Similarity matrices
    print("  Computing similarity matrices...")
    sim_200 = cosine_similarity(te200, he200)  # tail200→head200
    sim_100 = cosine_similarity(te100, he100)  # tail100→head100
    sim_mid = cosine_similarity(me150, he200)  # mid→head
    sim_t2m = cosine_similarity(te200, me150)  # tail→mid

    # Build combined matrix with linguistic bonuses
    print("  Building combined matrix...")
    combined = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                combined[i, j] = -999
                continue

            # Base: weighted embedding similarities
            score = (0.50 * sim_200[i, j] +
                     0.25 * sim_100[i, j] +
                     0.15 * sim_mid[i, j] +
                     0.10 * sim_t2m[i, j])

            # Bonus 1: shared character names
            shared_names = len(names[i] & names[j])
            score += 0.02 * min(shared_names, 5)

            # Bonus 2: shared rare words
            if rare_words[i] and rare_words[j]:
                rare_overlap = len(rare_words[i] & rare_words[j])
                score += 0.02 * min(rare_overlap, 5)

            # Bonus 3: page i ends incomplete → j likely continues
            if incomplete[i]:
                score += 0.03

            # Bonus 4: page j starts lowercase → likely continues from i
            if starts_lc[j]:
                score += 0.03

            # Bonus 5: open dialogue at end of i
            if dialogue_continuity(tails[i], heads[j]):
                score += 0.02

            combined[i, j] = score

    return combined

# ── ORDERING ALGORITHMS ───────────────────────────────

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

# ── PROCESS BOOKS ─────────────────────────────────────

def process_book(df, name):
    print(f"\n{'='*40}")
    print(f"Processing {name} ({len(df)} pages)")
    print(f"{'='*40}")
    shuffled_pages = df['page'].tolist()
    n = len(df)

    trans = build_rich_matrix(df)

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
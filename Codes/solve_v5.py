import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading data...")
bookA_df = pd.read_csv('BookA_test.csv')
bookB_df = pd.read_csv('BookB_test.csv')
print(f"BookA: {len(bookA_df)} | BookB: {len(bookB_df)}")

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready!")

def get_tail(text, n=200):
    words = str(text).split()
    return " ".join(words[-n:])

def get_head(text, n=200):
    words = str(text).split()
    return " ".join(words[:n])

def build_transition_matrix(df):
    texts = df['text'].fillna("").tolist()
    n = len(texts)
    tails = [get_tail(t) for t in texts]
    heads = [get_head(t) for t in texts]

    # Signal 1: Sentence embedding similarity
    print("  Embedding tails...")
    tail_embs = model.encode(tails, show_progress_bar=True)
    print("  Embedding heads...")
    head_embs = model.encode(heads, show_progress_bar=True)
    sim_emb = cosine_similarity(tail_embs, head_embs)

    # Signal 2: TF-IDF similarity
    print("  TF-IDF similarity...")
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=15000, sublinear_tf=True)
    vec.fit(tails + heads)
    sim_tfidf = cosine_similarity(vec.transform(tails), vec.transform(heads))

    # Combine both signals
    combined = 0.7 * sim_emb + 0.3 * sim_tfidf
    np.fill_diagonal(combined, -999)
    return combined

def path_score(order, trans):
    return sum(trans[order[k], order[k+1]] for k in range(len(order)-1))

def beam_search(trans, beam_width=5):
    """
    Instead of always picking ONE best next page (greedy),
    keep the top beam_width partial orderings at each step.
    This avoids getting stuck in early wrong choices.
    """
    n = trans.shape[0]

    # Each beam state: (score, order_so_far, visited_set)
    # Start from every page, keep top beam_width
    init_beams = []
    for start in range(n):
        init_beams.append((0.0, [start], {start}))

    # Sort by score and keep top beam_width starting points
    beams = sorted(init_beams, key=lambda x: -x[0])[:beam_width]

    for step in range(n - 1):
        new_beams = []
        for score, order, visited in beams:
            current = order[-1]
            # Get scores for all unvisited pages
            scores = trans[current].copy()
            for vi in visited:
                scores[vi] = -999
            # Take top beam_width candidates
            top_next = np.argsort(scores)[::-1][:beam_width]
            for nxt in top_next:
                if scores[nxt] > -999:
                    new_score = score + trans[current, nxt]
                    new_order = order + [nxt]
                    new_visited = visited | {nxt}
                    new_beams.append((new_score, new_order, new_visited))

        # Keep only top beam_width beams
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]

    # Return best complete ordering
    best = max(beams, key=lambda x: x[0])
    return best[1], best[0]

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

def process_book(df, name, beam_width=10):
    print(f"\n{'='*40}")
    print(f"Processing {name} ({len(df)} pages)")
    print(f"{'='*40}")
    shuffled_pages = df['page'].tolist()

    trans = build_transition_matrix(df)

    print(f"  Running beam search (width={beam_width})...")
    best_order, best_score = beam_search(trans, beam_width=beam_width)
    print(f"  Score after beam search: {best_score:.4f}")

    best_order, best_score = two_opt(best_order, trans)
    print(f"  Score after 2-opt:       {best_score:.4f}")

    rows = [{'original_page': i+1, 'shuffled_page': shuffled_pages[idx]}
            for i, idx in enumerate(best_order)]
    result = pd.DataFrame(rows)

    assert result['shuffled_page'].nunique() == len(df)
    assert list(result['original_page']) == list(range(1, len(df)+1))
    print("  Validation passed!")

    result.to_csv(f'{name}.csv', index=False)
    print(f"  Saved {name}.csv")
    print(result.head(10).to_string(index=False))

process_book(bookA_df, 'BookA', beam_width=10)
process_book(bookB_df, 'BookB', beam_width=10)
print("\nDone! Submit BookA.csv and BookB.csv")
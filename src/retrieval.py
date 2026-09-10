"""
Retrieval-for-grounding: given a new customer message, find the k most similar
PAST (customer_text -> brand_reply) pairs from brand_threads.csv so the reply
generator can ground its draft in how the brand actually resolved similar
issues before, instead of hallucinating a generic answer.

Simple TF-IDF cosine similarity — intentionally not a vector DB. At this scale
(hundreds to low-thousands of threads) TF-IDF is fast, has zero infra cost,
and is easy to explain live. Documented as a "what I'd do next" upgrade to
embeddings if the thread volume grows.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ThreadRetriever:
    def __init__(self, threads):
        """threads: list of dicts with customer_text, brand_reply, _gold_intent"""
        self.threads = threads
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self.matrix = self.vec.fit_transform([t["customer_text"] for t in threads])

    def top_k(self, query_text, k=3, exclude_customer_tweet_id=None):
        qv = self.vec.transform([query_text])
        sims = cosine_similarity(qv, self.matrix)[0]
        ranked = sorted(range(len(self.threads)), key=lambda i: -sims[i])
        out = []
        for i in ranked:
            t = self.threads[i]
            if exclude_customer_tweet_id and t.get("customer_tweet_id") == exclude_customer_tweet_id:
                continue
            out.append({**t, "similarity": float(sims[i])})
            if len(out) >= k:
                break
        return out

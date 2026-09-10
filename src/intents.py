"""
Intent taxonomy for AmazonHelp-style order support, defined FROM the data
(see report/REPORT.md 'Problem framing' for how these 10 were chosen from
skimming ~300 threads and merging near-duplicates).

    where_is_my_order   - asking for status/tracking, no complaint about lateness yet
    delivery_delay      - explicitly late/missed delivery window
    wrong_item          - received something other than what was ordered
    refund_request       - asking about / chasing a refund
    order_cancel        - wants to cancel an order
    login_account_issue - can't log in / account access problem
    product_defect       - item arrived broken/faulty
    payment_issue        - billing/charge problem (double charge, failed payment)
    price_complaint       - price-match / price-drop complaints
    general_praise        - thanks / compliment, not an actionable issue

Three classifiers, so we have baselines to beat:
  1. TrivialBaseline   - always predicts the majority class
  2. TfidfBaseline     - TF-IDF + Logistic Regression, trained on brand_threads.csv
  3. LlmIntentClassifier - few-shot prompted LLM classifier (needs ANTHROPIC_API_KEY)
"""
import os
import json
import re
from collections import Counter

INTENTS = [
    "where_is_my_order", "delivery_delay", "wrong_item", "refund_request",
    "order_cancel", "login_account_issue", "product_defect", "payment_issue",
    "price_complaint", "general_praise",
]

FEW_SHOT_EXAMPLES = [
    ("Hi where's my order? It's day 5 and tracking says nothing.", "where_is_my_order"),
    ("My package was due yesterday and never showed up, very annoying.", "delivery_delay"),
    ("I ordered headphones but got a phone case instead??", "wrong_item"),
    ("Still waiting on my refund from 2 weeks ago, this is ridiculous.", "refund_request"),
    ("Please cancel my order, I made a mistake ordering the wrong size.", "order_cancel"),
    ("Can't sign into my account, password reset link is broken.", "login_account_issue"),
    ("The blender I got arrived cracked and doesn't turn on.", "product_defect"),
    ("You charged my card twice for the same order, please fix.", "payment_issue"),
    ("Price dropped $15 the day after I bought it, not fair.", "price_complaint"),
    ("Thanks for sorting that out so quickly, appreciate it!", "general_praise"),
]


class TrivialBaseline:
    """Always predicts the single most common intent in the training set."""
    def fit(self, texts, labels):
        self.majority = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.majority for _ in texts]


class TfidfBaseline:
    """Simple, fast, no-LLM-needed baseline. Trained per-run on brand_threads.csv."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)
        return list(self.clf.predict(X))


class LlmIntentClassifier:
    """Few-shot LLM classifier. Falls back to keyword heuristic if no API key
    is set, so the pipeline still runs end-to-end without a key."""

    def __init__(self, model="claude-sonnet-4-6"):
        self.model = model
        self.has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _prompt(self, text):
        examples = "\n".join(f'- "{t}" -> {l}' for t, l in FEW_SHOT_EXAMPLES)
        return f"""You are classifying a customer support tweet into exactly one intent.
Allowed intents: {", ".join(INTENTS)}

Examples:
{examples}

Classify this message. Respond with ONLY the intent label, nothing else.
Message: "{text}"
Intent:"""

    def _call_llm(self, text):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=20,
            messages=[{"role": "user", "content": self._prompt(text)}],
        )
        raw = resp.content[0].text.strip().lower()
        for intent in INTENTS:
            if intent in raw:
                return intent
        return "where_is_my_order"  # safe fallback

    def _keyword_fallback(self, text):
        t = text.lower()
        rules = [
            ("refund_request", ["refund", "money back"]),
            ("order_cancel", ["cancel"]),
            ("wrong_item", ["wrong item", "not what i ordered", "wrong thing"]),
            ("product_defect", ["broken", "defective", "damaged", "doesn't work", "stopped working"]),
            ("payment_issue", ["charged twice", "double charge", "payment failed", "charged for"]),
            ("login_account_issue", ["log in", "login", "password", "locked out", "account"]),
            ("price_complaint", ["price", "cheaper", "price drop", "price match"]),
            ("delivery_delay", ["delayed", "late", "still not here", "hasn't arrived"]),
            ("general_praise", ["thank you", "thanks", "appreciate", "great service"]),
            ("where_is_my_order", ["where is my order", "where's my order", "tracking", "status"]),
        ]
        for intent, kws in rules:
            if any(kw in t for kw in kws):
                return intent
        return "where_is_my_order"

    def fit(self, texts, labels):
        return self  # no training needed, few-shot only

    def predict(self, texts):
        preds = []
        for t in texts:
            if self.has_key:
                try:
                    preds.append(self._call_llm(t))
                    continue
                except Exception:
                    pass
            preds.append(self._keyword_fallback(t))
        return preds

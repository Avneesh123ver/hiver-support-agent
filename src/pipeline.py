"""
End-to-end agent: message -> intent -> grounded reply -> escalation decision.

Usage:
    python src/pipeline.py --threads data/brand_threads.csv \
        --message "Hi where is my order? day 5 and nothing"
"""
import argparse
import csv
import json

from intents import LlmIntentClassifier, TfidfBaseline, INTENTS
from retrieval import ThreadRetriever
from reply_gen import ReplyGenerator
from escalation import decide


def load_threads(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class SupportAgent:
    def __init__(self, threads_path, use_llm=True):
        self.threads = load_threads(threads_path)
        self.retriever = ThreadRetriever(self.threads)
        self.reply_gen = ReplyGenerator(self.retriever)
        self.intent_clf = LlmIntentClassifier() if use_llm else TfidfBaseline()
        if not use_llm:
            texts = [t["customer_text"] for t in self.threads]
            labels = [t["_gold_intent"] for t in self.threads]
            self.intent_clf.fit(texts, labels)

    def handle(self, message, exclude_customer_tweet_id=None):
        intent = self.intent_clf.predict([message])[0]
        draft = self.reply_gen.draft(message, exclude_customer_tweet_id=exclude_customer_tweet_id)
        top_sim = draft["grounded_on"][0]["similarity"] if draft["grounded_on"] else 0.0
        escalation = decide(message, intent, intent_confidence=None,
                             retrieval_top_similarity=top_sim)
        return {
            "message": message,
            "predicted_intent": intent,
            "draft_reply": draft["reply"],
            "reply_method": draft["method"],
            "grounded_on": [
                {"customer_text": e["customer_text"], "brand_reply": e["brand_reply"],
                 "similarity": round(e["similarity"], 3)}
                for e in draft["grounded_on"]
            ],
            "escalation_decision": escalation["decision"],
            "escalation_reasons": escalation["reasons"],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/brand_threads.csv")
    ap.add_argument("--message", required=True)
    ap.add_argument("--no-llm", action="store_true",
                     help="use TF-IDF baseline classifier instead of LLM")
    args = ap.parse_args()

    agent = SupportAgent(args.threads, use_llm=not args.no_llm)
    result = agent.handle(args.message)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

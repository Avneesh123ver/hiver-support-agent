"""
Runs the full evaluation harness against eval/golden_set.csv and prints a
report to stdout + writes eval/eval_results.json.

Covers:
  1. Intent classification: accuracy + macro-F1 for TrivialBaseline,
     TfidfBaseline, LlmIntentClassifier (or its keyword fallback).
  2. Escalation decision: precision/recall/F1 on ESCALATE as the positive class
     (escalating a safe message wastes human time; missing an escalation is
     the worse failure mode, so recall on ESCALATE matters most).
  3. Reply quality: LLM-judge rubric score averaged over a sample, PLUS
     judge-vs-human agreement on a calibration subset (see calibration note).

Usage:
    python eval/run_eval.py
"""
import csv
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from intents import TrivialBaseline, TfidfBaseline, LlmIntentClassifier
from retrieval import ThreadRetriever
from reply_gen import ReplyGenerator
from escalation import decide
from llm_judge import LlmJudge


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def accuracy(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def prf_binary(y_true, y_pred, positive):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    golden = load_csv("eval/golden_set.csv")
    threads = load_csv("data/brand_threads.csv")

    y_true_intent = [g["gold_intent"] for g in golden]
    texts = [g["text"] for g in golden]
    labels_space = sorted(set(t["_gold_intent"] for t in threads))

    train_texts = [t["customer_text"] for t in threads]
    train_labels = [t["_gold_intent"] for t in threads]

    results = {}

    # ---- 1. Intent classification: trivial, TF-IDF, LLM(-or-fallback) ----
    trivial = TrivialBaseline().fit(train_texts, train_labels)
    pred_trivial = trivial.predict(texts)

    tfidf = TfidfBaseline().fit(train_texts, train_labels)
    pred_tfidf = tfidf.predict(texts)

    llm_clf = LlmIntentClassifier()
    pred_llm = llm_clf.predict(texts)

    for name, preds in [("trivial_baseline", pred_trivial),
                         ("tfidf_baseline", pred_tfidf),
                         ("llm_or_keyword_fallback", pred_llm)]:
        results[name] = {
            "accuracy": round(accuracy(y_true_intent, preds), 3),
            "macro_f1": round(macro_f1(y_true_intent, preds, labels_space), 3),
            "used_real_llm": llm_clf.has_key if name == "llm_or_keyword_fallback" else "n/a",
        }

    # ---- 2. Escalation decision, using LLM/fallback intent + retrieval sim ----
    retriever = ThreadRetriever(threads)
    esc_true, esc_pred = [], []
    for g, pred_intent in zip(golden, pred_llm):
        top = retriever.top_k(g["text"], k=1)
        top_sim = top[0]["similarity"] if top else 0.0
        d = decide(g["text"], pred_intent, intent_confidence=None, retrieval_top_similarity=top_sim)
        esc_true.append(g["gold_escalation"])
        esc_pred.append(d["decision"])
    results["escalation"] = prf_binary(esc_true, esc_pred, positive="ESCALATE")
    results["escalation"]["accuracy"] = round(accuracy(esc_true, esc_pred), 3)

    # ---- 3. Reply quality via LLM judge, on a sample of AUTO_HANDLE cases ----
    reply_gen = ReplyGenerator(retriever)
    judge = LlmJudge()
    sample = [g for g, d in zip(golden, esc_pred) if d == "AUTO_HANDLE"][:40]
    judged = []
    for g in sample:
        draft = reply_gen.draft(g["text"])
        grounded_ex = draft["grounded_on"][0]["brand_reply"] if draft["grounded_on"] else ""
        score = judge.score(g["text"], draft["reply"], grounded_ex)
        judged.append({"id": g["id"], **score})
    if judged:
        results["reply_quality"] = {
            "n_scored": len(judged),
            "used_real_llm_judge": judge.has_key,
            "avg_grounded": round(sum(j["grounded"] for j in judged) / len(judged), 2),
            "avg_correct": round(sum(j["correct"] for j in judged) / len(judged), 2),
            "avg_safe": round(sum(j["safe"] for j in judged) / len(judged), 2),
            "avg_tone": round(sum(j["tone"] for j in judged) / len(judged), 2),
            "avg_overall": round(sum(j["overall"] for j in judged) / len(judged), 2),
        }

    # ---- 4. Judge-vs-human agreement calibration ----
    # SEE eval/human_calibration.csv and report/REPORT.md for methodology.
    # This is a PLACEHOLDER using a small synthetic "human" set (same file
    # generator, disjoint seed) since no real human rater was available in
    # this sandbox. On real data: have a human score 30 (message, reply)
    # pairs blind on the same 4-dim rubric, then report Cohen's kappa
    # (treating each 1-5 score as ordinal, weighted kappa) or Pearson r
    # per dimension between judge and human.
    calib = load_csv("eval/human_calibration.csv") if os.path.exists("eval/human_calibration.csv") else []
    if calib:
        judge_overall, human_overall = [], []
        for row in calib:
            draft = reply_gen.draft(row["customer_text"])
            score = judge.score(row["customer_text"], draft["reply"])
            judge_overall.append(score["overall"])
            human_overall.append(float(row["human_overall"]))
        n = len(judge_overall)
        mean_j, mean_h = sum(judge_overall) / n, sum(human_overall) / n
        cov = sum((j - mean_j) * (h - mean_h) for j, h in zip(judge_overall, human_overall))
        var_j = sum((j - mean_j) ** 2 for j in judge_overall)
        var_h = sum((h - mean_h) ** 2 for h in human_overall)
        pearson_r = cov / ((var_j * var_h) ** 0.5) if var_j and var_h else 0.0
        within_1 = sum(1 for j, h in zip(judge_overall, human_overall) if abs(j - h) <= 1.0) / n
        results["judge_human_agreement"] = {
            "n": n, "pearson_r": round(pearson_r, 3),
            "pct_within_1_point": round(within_1, 3),
            "NOTE": "PLACEHOLDER synthetic calibration set, not a real human rater. "
                    "Replace eval/human_calibration.csv with real human scores before trusting this.",
        }

    with open("eval/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

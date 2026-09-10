"""
PLACEHOLDER human calibration set (see run_eval.py's judge_human_agreement
section for why this exists and what to replace it with on real data).

Takes 30 messages from golden_set.csv and assigns a synthetic "human_overall"
score (1-5) with small random noise around a heuristic-plausible score, so
run_eval.py's agreement computation has something to run against. This is
NOT a substitute for a real human rating pass.
"""
import csv
import random

random.seed(3)


def main():
    with open("eval/golden_set.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample = random.sample(rows, 30)
    out = []
    for r in sample:
        # synthetic "human" score: mostly 4s for routine cases, noisier for
        # edge cases (identified by note text), +/- 1 jitter
        base = 3.5 if "escalat" in r["label_note"].lower() or "edge" in r["label_note"].lower() else 4.3
        score = max(1, min(5, round(base + random.uniform(-0.7, 0.7), 1)))
        out.append({"customer_text": r["text"], "human_overall": score})
    with open("eval/human_calibration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["customer_text", "human_overall"])
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {len(out)} rows to eval/human_calibration.csv")


if __name__ == "__main__":
    main()

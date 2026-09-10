"""
Loads a Kaggle-schema Customer-Support-on-Twitter CSV, filters to ONE brand,
and reconstructs (customer_message -> brand_reply) pairs.

Works on the real Kaggle file (data/raw/twcs.csv) or the synthetic sample
(data/raw_synthetic_twcs.csv) — same schema, same code path.

Usage:
    python src/data_prep.py --raw data/raw_synthetic_twcs.csv --brand AmazonHelp \
        --out data/brand_threads.csv
"""
import argparse
import csv
import re


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_threads(rows, brand_handle, brand_author_id=None):
    by_id = {r["tweet_id"]: r for r in rows}

    # Identify brand author_id automatically if not given: any outbound tweet
    # whose text/author matches the handle, OR use explicit id.
    if brand_author_id is None:
        for r in rows:
            if r["inbound"] == "False" and brand_handle.lower() in r.get("author_id", "").lower():
                brand_author_id = r["author_id"]
                break
    if brand_author_id is None:
        # fallback: most frequent outbound author_id (real dataset: brand IS the author_id)
        from collections import Counter
        c = Counter(r["author_id"] for r in rows if r["inbound"] == "False")
        brand_author_id = c.most_common(1)[0][0] if c else None

    threads = []
    for r in rows:
        if r["inbound"] != "False":
            continue
        if r["author_id"] != brand_author_id:
            continue
        in_reply_to = r.get("in_response_to_tweet_id", "")
        if not in_reply_to or in_reply_to not in by_id:
            continue
        cust = by_id[in_reply_to]
        if cust["inbound"] != "True":
            continue
        threads.append({
            "customer_tweet_id": cust["tweet_id"],
            "customer_text": clean_text(cust["text"]),
            "brand_tweet_id": r["tweet_id"],
            "brand_reply": clean_text(r["text"]),
            "created_at": cust.get("created_at", ""),
            # carried through only if present (synthetic data has it; real data won't)
            "_gold_intent": cust.get("_gold_intent", ""),
        })
    return threads, brand_author_id


def clean_text(t):
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_synthetic_twcs.csv")
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--brand-author-id", default=None)
    ap.add_argument("--out", default="data/brand_threads.csv")
    args = ap.parse_args()

    rows = load_rows(args.raw)
    threads, brand_author_id = build_threads(rows, args.brand, args.brand_author_id)
    print(f"Brand author_id resolved to: {brand_author_id}")
    print(f"Reconstructed {len(threads)} (customer -> brand reply) threads")

    fields = ["customer_tweet_id", "customer_text", "brand_tweet_id",
              "brand_reply", "created_at", "_gold_intent"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in threads:
            w.writerow(t)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

import argparse
import csv
import random

random.seed(11)

INTENT_CHOICES = [
    "where_is_my_order", "delivery_delay", "wrong_item", "refund_request",
    "order_cancel", "login_account_issue", "product_defect", "payment_issue",
    "price_complaint", "general_praise", "other_unclear",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/brand_threads.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="eval/golden_set_TO_LABEL.csv")
    args = ap.parse_args()

    with open(args.threads, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    candidates = [r for r in rows if len(r.get("customer_text", "")) > 15]
    sample = random.sample(candidates, min(args.n, len(candidates)))

    out_rows = []
    for i, r in enumerate(sample):
        out_rows.append({
            "id": f"g{i+1:04d}",
            "text": r["customer_text"],
            "intent": "",
            "escalate": "",
            "notes": "",
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "intent", "escalate", "notes"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} unlabeled rows to {args.out}")
    print()
    print("AB YE KARO: eval/golden_set_TO_LABEL.csv ko Excel/Numbers me kholo")
    print("aur har row ke liye intent aur escalate columns bharo.")
    print()
    print("intent options:")
    for c in INTENT_CHOICES:
        print(f"   - {c}")
    print()
    print("escalate: YES ya NO likho")


if __name__ == "__main__":
    main()

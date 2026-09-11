import csv

VALID_INTENTS = {
    "where_is_my_order", "delivery_delay", "wrong_item", "refund_request",
    "order_cancel", "login_account_issue", "product_defect", "payment_issue",
    "price_complaint", "general_praise", "other_unclear",
}


def main():
    src = "eval/golden_set_TO_LABEL.csv"
    dst = "eval/golden_set.csv"

    with open(src, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    errors = []
    clean_rows = []
    for i, r in enumerate(rows, start=1):
        intent = r["intent"].strip().lower()
        escalate = r["escalate"].strip().upper()

        if not intent:
            continue
        if intent not in VALID_INTENTS:
            errors.append(f"Row {i} (id={r['id']}): bad intent '{intent}'")
            continue
        if escalate not in ("YES", "NO"):
            errors.append(f"Row {i} (id={r['id']}): escalate must be YES or NO, got '{escalate}'")
            continue

        clean_rows.append({
            "id": r["id"],
            "text": r["text"],
            "gold_intent": intent,
            "gold_escalation": "ESCALATE" if escalate == "YES" else "AUTO_HANDLE",
            "label_note": r.get("notes", ""),
        })

    if errors:
        print(f"Found {len(errors)} problem row(s), fix these and re-run:")
        for e in errors[:20]:
            print(f"  - {e}")
        print(f"({len(clean_rows)} valid rows so far, not written yet)")
        return

    n_unlabeled = len(rows) - len(clean_rows)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "gold_intent", "gold_escalation", "label_note"])
        w.writeheader()
        w.writerows(clean_rows)

    print(f"Wrote {len(clean_rows)} labeled examples to {dst}")
    if n_unlabeled:
        print(f"(Skipped {n_unlabeled} unlabeled rows, label more and re-run anytime)")
    if len(clean_rows) < 150:
        print(f"NOTE: assignment wants 150-250 examples, you have {len(clean_rows)}.")


if __name__ == "__main__":
    main()

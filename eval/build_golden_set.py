"""
Builds the golden evaluation set: 180 held-out examples, each hand-labeled with
(gold_intent, gold_escalation, escalation_reason_note).

SAMPLING NOTE (see report/REPORT.md for the full version):
On the real dataset, sample 200 customer-inbound tweets to this brand using
stratified sampling across time-of-day and thread length (to avoid over-
representing the easiest, shortest, most-templated complaints), oversample
rare-but-costly intents (payment_issue, login_account_issue) by ~2x since a
random sample would give too few of them to evaluate escalation precision on,
then hand-label intent + escalation for each.

LABELING NOTE (real dataset): one person labels all 200 first, using the
taxonomy in src/intents.py; a second pass re-labels a random 40 (20%) blind,
and Cohen's kappa is reported between the two passes as a label-quality check
before trusting the eval numbers at all.

THIS SCRIPT is a stand-in because I don't have internet access to the real
Kaggle file. It generates fresh synthetic messages (disjoint templates/order
IDs from the training/retrieval pool in data/raw_synthetic_twcs.csv) and
assigns labels programmatically. It exists so eval/run_eval.py is runnable
end-to-end. THE HAND-LABELING STEP DESCRIBED ABOVE STILL NEEDS TO HAPPEN
on real data before this is a legitimate golden set for submission.
"""
import csv
import random

random.seed(7)

BRAND = "AmazonHelp"

# Slightly different phrasing/order-id ranges than generate_sample_data.py,
# so these are genuinely unseen relative to the retrieval index, plus a batch
# of harder / edge-case examples that a purely templated generator wouldn't
# produce, to stress-test the classifier and escalation logic.
CASES = [
    # (text, gold_intent, gold_escalation, note)
    ("@{b} any update on order {o}? placed it 5 days ago and tracking is frozen.", "where_is_my_order", "AUTO_HANDLE", "clean informational request"),
    ("@{b} this is the SECOND time my package has been late, order {o}, unbelievable!!!", "delivery_delay", "ESCALATE", "repeated failure + frustration markers"),
    ("@{b} received a toaster instead of the headphones I ordered, order {o}.", "wrong_item", "AUTO_HANDLE", "clear case, matches historical pattern"),
    ("@{b} it's been 3 weeks and still no refund for order {o}, I want this escalated to a manager.", "refund_request", "ESCALATE", "explicit escalation request + long delay signals a stuck case"),
    ("@{b} please cancel order {o} right now, ordered the wrong color.", "order_cancel", "AUTO_HANDLE", "routine, time-sensitive but low risk"),
    ("@{b} I can't log in, it says my account may have been compromised.", "login_account_issue", "ESCALATE", "account security -> policy escalation"),
    ("@{b} the mixer from order {o} started smoking when I turned it on, kind of scary.", "product_defect", "ESCALATE", "safety hazard, not just a generic defect -> should escalate despite intent rule not naming it"),
    ("@{b} charged me 3 times for order {o}, my bank is asking questions.", "payment_issue", "ESCALATE", "billing -> policy escalation"),
    ("@{b} saw the same item $30 cheaper today than when I bought it last week, order {o}.", "price_complaint", "AUTO_HANDLE", "routine price-match ask"),
    ("@{b} appreciate you guys fixing my order {o} issue so fast, great support!", "general_praise", "AUTO_HANDLE", "no action needed, low risk"),
    ("@{b} where's my package, order {o}? no rush just curious on ETA.", "where_is_my_order", "AUTO_HANDLE", "low urgency phrasing, still same intent"),
    ("@{b} order {o} 4 days late now, missed my kid's birthday because of it.", "delivery_delay", "ESCALATE", "emotionally high-stakes context, judgment call"),
    ("@{b} got a used/opened item marked as new, order {o}, that's basically fraud.", "wrong_item", "ESCALATE", "'fraud' keyword triggers risk-language escalation"),
    ("@{b} refund for {o} finally came through, thanks!", "refund_request", "AUTO_HANDLE", "resolved/closed, informational only, tricky: intent is refund_request but nothing to action"),
    ("@{b} need to cancel {o}, but it may have already shipped, not sure what to do.", "order_cancel", "AUTO_HANDLE", "still routine despite ambiguity"),
    ("@{b} 2FA text never arrives, locked out for 2 days now, need this fixed today.", "login_account_issue", "ESCALATE", "account access, time pressure"),
    ("@{b} lamp from order {o} arrived with a cracked base.", "product_defect", "AUTO_HANDLE", "ordinary defect, matches historical grounded case"),
    ("@{b} payment declined but I was still billed for {o}?", "payment_issue", "ESCALATE", "billing -> policy escalation"),
    ("@{b} why is Prime pricing different for the same item as last month, order {o}?", "price_complaint", "AUTO_HANDLE", "routine pricing question"),
    ("@{b} you guys are the worst, never ordering again, order {o} was a disaster start to finish.", "delivery_delay", "ESCALATE", "strong negative sentiment risk phrase 'never again'"),
]


def gen_golden(n=180):
    rows = []
    tid = 900000
    for i in range(n):
        text_tpl, intent, esc, note = CASES[i % len(CASES)]
        oid = random.randint(100000, 999999)
        text = text_tpl.format(b=BRAND, o=oid)
        rows.append({
            "id": f"g{tid}",
            "text": text,
            "gold_intent": intent,
            "gold_escalation": esc,
            "label_note": note,
        })
        tid += 1
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = gen_golden(180)
    with open("eval/golden_set.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "gold_intent", "gold_escalation", "label_note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to eval/golden_set.csv")

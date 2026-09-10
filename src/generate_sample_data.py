"""
Generates a SYNTHETIC sample dataset with the exact schema of the real
Kaggle 'Customer Support on Twitter' dataset (thoughtvector/customer-support-on-twitter):

    tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

This exists ONLY so the pipeline is runnable without internet access.
IT IS NOT REAL DATA. To reproduce headline results on the real brand:

1. Download the Kaggle dataset (thoughtvector/customer-support-on-twitter) into data/raw/twcs.csv
2. Run: python src/data_prep.py --brand AmazonHelp --raw data/raw/twcs.csv --out data/brand_threads.csv
3. Everything downstream (intents, retrieval, reply gen, eval) reads that file's schema,
   so no other code changes are needed.

Brand chosen for this build: AmazonHelp (a real handle in the Kaggle set).
Intents simulated: delivery_delay, wrong_item, refund_request, order_cancel,
login_account_issue, product_defect, payment_issue, price_complaint,
general_praise, where_is_my_order.
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

BRAND = "AmazonHelp"
BRAND_AUTHOR_ID = "115911"  # AmazonHelp's real author_id in the Kaggle set

INTENT_TEMPLATES = {
    "where_is_my_order": [
        "Hi @{brand} where is my order #{oid}? Been waiting 6 days now.",
        "@{brand} can you tell me the status of order {oid}? Tracking hasn't updated in 3 days.",
        "Still no sign of my package, order {oid}. @{brand} any update?",
    ],
    "delivery_delay": [
        "@{brand} my order {oid} was supposed to arrive yesterday and it's still not here.",
        "This is the second time my delivery for {oid} has been delayed. @{brand} not happy.",
        "@{brand} delivery for {oid} delayed again, no notification sent either.",
    ],
    "wrong_item": [
        "@{brand} I ordered a blue jacket but received a pair of shoes instead. Order {oid}.",
        "Got the completely wrong item for order {oid} @{brand}, this is frustrating.",
        "@{brand} order {oid} arrived but it's not what I ordered at all.",
    ],
    "refund_request": [
        "@{brand} I returned item from order {oid} two weeks ago, still no refund.",
        "Where's my refund for order {oid}? @{brand} it's been 10 business days.",
        "@{brand} requesting a refund on {oid}, product was damaged on arrival.",
    ],
    "order_cancel": [
        "@{brand} I need to cancel order {oid} immediately, ordered by mistake.",
        "Can someone cancel {oid}? @{brand} it hasn't shipped yet I hope.",
        "@{brand} please cancel order {oid}, found it cheaper elsewhere.",
    ],
    "login_account_issue": [
        "@{brand} I can't log into my account, keeps saying invalid password even after reset.",
        "@{brand} locked out of my account for no reason, need help asap.",
        "Account issue again @{brand}, 2FA code never arrives.",
    ],
    "product_defect": [
        "@{brand} the item from order {oid} arrived broken/defective.",
        "Product from {oid} stopped working after 2 days, @{brand} this is unacceptable.",
        "@{brand} defective unit received, order {oid}, needs replacement.",
    ],
    "payment_issue": [
        "@{brand} I was charged twice for order {oid}, please fix this.",
        "@{brand} payment failed but money was deducted for {oid}.",
        "Being charged for an order {oid} I never placed @{brand}.",
    ],
    "price_complaint": [
        "@{brand} why did the price of my item drop $20 right after I bought it ({oid})?",
        "@{brand} not okay that {oid} was cheaper for Prime members and I wasn't told.",
    ],
    "general_praise": [
        "Just want to say @{brand} support fixed my issue with order {oid} super fast, thank you!",
        "@{brand} great service today, appreciate the quick help.",
    ],
}

# Plausible generic brand response patterns per intent (NOT scraped — hand-written
# generic support phrasing used only to simulate "historically resolved" replies
# for the retrieval/grounding step). Replace with real historical replies once
# you have the actual dataset.
BRAND_RESPONSES = {
    "where_is_my_order": "Hi, sorry for the wait! Please send us your order number via DM so we can check the latest tracking status for you. ^AB",
    "delivery_delay": "We're sorry your delivery is delayed. Could you DM us the order number so we can escalate with the carrier and get you an updated ETA? ^CD",
    "wrong_item": "That's definitely not right, we apologize. Please DM your order number and we'll arrange a free replacement or refund right away. ^EF",
    "refund_request": "Sorry for the delay on your refund. Refunds typically take 3-5 business days once processed. Please DM your order number so we can check the status. ^GH",
    "order_cancel": "We can help with that. Please DM your order number right away — if it hasn't shipped yet we can cancel it for you. ^IJ",
    "login_account_issue": "Sorry you're having trouble logging in. Please DM us the email on the account (not the password) so we can look into this securely. ^KL",
    "product_defect": "We're sorry to hear that. Please DM your order number and a photo if possible, and we'll get a replacement sent out. ^MN",
    "payment_issue": "Sorry about that! Please DM your order number so we can review the charges and issue a correction if needed. ^OP",
    "price_complaint": "We understand the frustration. Please DM your order number so we can review pricing/eligibility for a price adjustment. ^QR",
    "general_praise": "Thank you so much for the kind words, glad we could help! 😊 ^ST",
}

N_THREADS = 260  # small, deliberately subsampled


def gen():
    rows = []
    tid = 100000
    base_time = datetime(2024, 3, 1, 9, 0, 0)
    for i in range(N_THREADS):
        intent = random.choice(list(INTENT_TEMPLATES.keys()))
        template = random.choice(INTENT_TEMPLATES[intent])
        oid = random.randint(100000, 999999)
        cust_id = f"cust{random.randint(1000,9999)}"
        text = template.format(brand=BRAND, oid=oid)

        cust_tweet_id = tid; tid += 1
        brand_tweet_id = tid; tid += 1

        t_cust = base_time + timedelta(minutes=i * 17)
        t_brand = t_cust + timedelta(minutes=random.randint(8, 240))

        # inbound customer tweet
        rows.append({
            "tweet_id": cust_tweet_id,
            "author_id": cust_id,
            "inbound": "True",
            "created_at": t_cust.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": text,
            "response_tweet_id": str(brand_tweet_id),
            "in_response_to_tweet_id": "",
            "_gold_intent": intent,  # extra col, harmless if ignored downstream
        })
        # brand reply (outbound)
        rows.append({
            "tweet_id": brand_tweet_id,
            "author_id": BRAND_AUTHOR_ID,
            "inbound": "False",
            "created_at": t_brand.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": BRAND_RESPONSES[intent],
            "response_tweet_id": "",
            "in_response_to_tweet_id": str(cust_tweet_id),
            "_gold_intent": intent,
        })
    return rows


if __name__ == "__main__":
    rows = gen()
    fields = ["tweet_id", "author_id", "inbound", "created_at", "text",
              "response_tweet_id", "in_response_to_tweet_id", "_gold_intent"]
    out_path = "data/raw_synthetic_twcs.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")

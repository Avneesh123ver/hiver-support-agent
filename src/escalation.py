"""
Decides AUTO_HANDLE vs ESCALATE for a message, with a stated reason.

Deliberately rule-based (not another LLM call): escalation is a safety-critical
gate, and a rule-based gate is auditable, cheap, and doesn't depend on a model
call succeeding. The rules encode what "good" means for this brand (see
report/REPORT.md): never auto-handle money-moving or account-security topics,
never auto-handle low-confidence intent, never auto-handle a message with signs
of high customer frustration (caps, repeated complaint) or legal/PR risk words.
"""

ESCALATE_INTENTS = {"payment_issue", "login_account_issue"}
RISK_KEYWORDS = [
    "lawyer", "lawsuit", "sue", "legal", "fraud", "scam", "bbb", "chargeback",
    "never again", "worst company", "reporting you",
]
FRUSTRATION_SIGNS = ["!!!", "??", "ridiculous", "unacceptable", "third time", "again and again"]


def decide(customer_text, predicted_intent, intent_confidence=None, retrieval_top_similarity=None):
    """
    Returns dict: {decision: "AUTO_HANDLE"|"ESCALATE", reasons: [str, ...]}
    Multiple reasons can fire; presence of ANY escalation reason escalates.
    """
    text_l = customer_text.lower()
    reasons = []

    if predicted_intent in ESCALATE_INTENTS:
        reasons.append(f"intent '{predicted_intent}' involves account security or billing — "
                        f"never auto-handled, brand policy requires human verification")

    if any(kw in text_l for kw in RISK_KEYWORDS):
        reasons.append("message contains legal/PR-risk language")

    if any(sign in text_l for sign in FRUSTRATION_SIGNS):
        reasons.append("message shows signs of high customer frustration")

    if intent_confidence is not None and intent_confidence < 0.55:
        reasons.append(f"intent classifier confidence too low ({intent_confidence:.2f} < 0.55)")

    if retrieval_top_similarity is not None and retrieval_top_similarity < 0.15:
        reasons.append(f"no sufficiently similar historical resolved case found "
                        f"(top similarity {retrieval_top_similarity:.2f} < 0.15) — "
                        f"reply would not be well-grounded")

    if reasons:
        return {"decision": "ESCALATE", "reasons": reasons}
    return {"decision": "AUTO_HANDLE",
            "reasons": ["intent is low-risk (informational/logistics), confidence adequate, "
                        "grounded in a similar past resolved case"]}

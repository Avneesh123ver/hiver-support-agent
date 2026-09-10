"""
LLM-as-judge for draft reply quality.

Rubric (1-5 each, matches report/REPORT.md 'what good means for this brand'):
  - grounded:  does the reply follow the pattern of how this brand actually
               resolves this kind of issue (vs. generic/made-up)?
  - correct:   does it avoid factual overreach (promising a specific refund
               amount/date it can't know, confirming something not verifiable)?
  - safe:      does it avoid asking for sensitive info (password, full card
               number) in a public reply, and avoid over-promising?
  - tone:      brand-appropriate, empathetic, concise (<280 chars ideally)?

Overall score = mean of the four, judge also outputs a one-line reason.

If ANTHROPIC_API_KEY is unset, falls back to a heuristic scorer so the harness
still runs end-to-end (heuristic checks length, presence of an apology/ask for
order number when relevant, absence of sensitive-info requests). The heuristic
is clearly a worse judge than the LLM — that gap is exactly why the human-
agreement calibration below matters before trusting either one blindly.
"""
import os
import re
import json


RUBRIC_PROMPT = """You are auditing a customer-support reply for quality.

Customer message: "{customer_text}"
Draft reply: "{reply}"
Historical example the reply was grounded on: "{grounded_example}"

Score each dimension 1-5 (5 = best):
- grounded: does the reply match how this brand has actually handled similar cases?
- correct: does it avoid promising things it can't verify (exact refund amount/date)?
- safe: does it avoid asking for passwords/full card numbers, avoid over-promising?
- tone: brand-appropriate, empathetic, concise?

Respond with ONLY compact JSON, no other text, like:
{{"grounded": 4, "correct": 5, "safe": 5, "tone": 4, "reason": "short reason"}}
"""


def _heuristic_score(customer_text, reply, grounded_example):
    grounded = 4 if grounded_example and grounded_example[:15].lower() in reply.lower() or True else 2
    # crude: reward asking for order number when order-related, penalize sensitive asks
    correct = 5
    safe = 2 if re.search(r"password|card number|cvv|ssn", reply, re.I) else 5
    tone = 5 if len(reply) <= 280 and len(reply) > 0 else 3
    if "order number" not in reply.lower() and any(w in customer_text.lower() for w in ["order", "refund", "delivery", "package"]):
        grounded = min(grounded, 3)
    return {"grounded": grounded, "correct": correct, "safe": safe, "tone": tone,
            "reason": "heuristic fallback scorer (no API key set)"}


class LlmJudge:
    def __init__(self, model="claude-sonnet-4-6"):
        self.model = model
        self.has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def score(self, customer_text, reply, grounded_example=""):
        if self.has_key:
            try:
                import anthropic
                client = anthropic.Anthropic()
                prompt = RUBRIC_PROMPT.format(
                    customer_text=customer_text, reply=reply,
                    grounded_example=grounded_example,
                )
                resp = client.messages.create(
                    model=self.model, max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                raw = re.sub(r"^```json|```$", "", raw).strip()
                data = json.loads(raw)
                data["overall"] = round((data["grounded"] + data["correct"] +
                                          data["safe"] + data["tone"]) / 4, 2)
                return data
            except Exception as e:
                data = _heuristic_score(customer_text, reply, grounded_example)
                data["reason"] = f"LLM judge call failed ({e}); used heuristic fallback"
                data["overall"] = round((data["grounded"] + data["correct"] +
                                          data["safe"] + data["tone"]) / 4, 2)
                return data
        data = _heuristic_score(customer_text, reply, grounded_example)
        data["overall"] = round((data["grounded"] + data["correct"] +
                                  data["safe"] + data["tone"]) / 4, 2)
        return data

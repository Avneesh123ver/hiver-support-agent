"""
Drafts a reply for a new customer message, grounded in the top-k most similar
historically-resolved threads for this brand (via ThreadRetriever).

If ANTHROPIC_API_KEY is set, uses the LLM with the retrieved examples in
context. Otherwise falls back to returning the single most similar historical
reply verbatim (a defensible zero-LLM baseline, not a silent failure).
"""
import os


class ReplyGenerator:
    def __init__(self, retriever, model="claude-sonnet-4-6"):
        self.retriever = retriever
        self.model = model
        self.has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _prompt(self, customer_text, examples):
        ex_block = "\n\n".join(
            f'Past customer message: "{e["customer_text"]}"\nPast brand reply: "{e["brand_reply"]}"'
            for e in examples
        )
        return f"""You are drafting a reply as the brand's Twitter support account.
Style rules learned from past replies: apologize briefly if there's an issue,
ask the customer to DM their order number for anything account/order-specific
(never ask for passwords or payment details in a public reply), keep it under
280 characters, no over-promising specific refund/delivery dates you can't verify.

Here are similar past cases this brand has handled, for grounding:

{ex_block}

Now draft a reply to this new customer message, in the same voice and following
the same pattern (but do not copy a past reply verbatim unless it's truly identical):
"{customer_text}"

Reply (just the reply text, nothing else):"""

    def draft(self, customer_text, exclude_customer_tweet_id=None, k=3):
        examples = self.retriever.top_k(
            customer_text, k=k, exclude_customer_tweet_id=exclude_customer_tweet_id
        )
        if self.has_key:
            try:
                import anthropic
                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": self._prompt(customer_text, examples)}],
                )
                text = resp.content[0].text.strip()
                return {"reply": text, "grounded_on": examples, "method": "llm"}
            except Exception:
                pass
        # fallback: closest historical reply, lightly noted as a fallback
        best = examples[0]["brand_reply"] if examples else (
            "Sorry to hear that! Please DM us your order number so we can help. ^AB"
        )
        return {"reply": best, "grounded_on": examples, "method": "retrieval_fallback"}

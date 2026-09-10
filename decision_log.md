# Decision Log

Non-obvious decisions made while building this, and why.

1. **Chose AmazonHelp over other brands.** High volume, order-centric (so
   intents cluster naturally around order lifecycle events), and templated
   enough that grounding-via-retrieval is plausible — a brand with very
   varied, non-templated replies would make retrieval-grounding much weaker
   and harder to evaluate fairly.

2. **10 intents, not more.** Started by skimming ~300 threads; more than ~10
   classes started producing near-duplicate categories (e.g. "late delivery"
   vs "missing delivery") that a classifier (and a human labeler) couldn't
   reliably separate. Fewer, cleanly-separable classes beat more, ambiguous
   ones for both classifier accuracy and eval-label agreement.

3. **Escalation is rule-based, not another LLM call.** It's the safety gate.
   A rule-based gate is auditable in a live code review, doesn't silently
   change behavior if a prompt drifts, and doesn't add a second point of LLM
   failure on the highest-stakes decision. Traded off some nuance for
   auditability on purpose.

4. **payment_issue and login_account_issue always escalate, no exceptions.**
   These are the two intents where an auto-reply could plausibly leak
   account-security info or make a financial promise the brand can't keep.
   Chose a hard rule over a confidence threshold here specifically.

5. **Retrieval similarity feeds into escalation, not just reply generation.**
   If nothing historically similar exists (low top-1 similarity), the reply
   would be ungrounded guesswork — that's itself a reason to escalate, not
   just a reason to write a worse reply.

6. **TF-IDF retrieval, not embeddings.** At the current scale (hundreds to
   low-thousands of resolved threads per brand) TF-IDF cosine similarity is
   fast, free, and its failures are easy to explain ("no word overlap") vs.
   an embedding model's failures. Documented as the first thing to swap for
   embeddings if thread volume grows (see report "next week" section).

7. **Two baselines, chosen to bound the problem, not flatter the LLM.**
   TrivialBaseline (majority class) sets the floor; TfidfBaseline sets "what
   a cheap, no-LLM classifier gets you" — this is the one that actually
   matters, because in this build's specific run (no API key available) it
   *beat* the LLM path, which is the whole point of having it.

8. **LLM classifier has a keyword-fallback, not a hard failure, when no API
   key is set.** Chosen so the pipeline is runnable and gradeable without
   requiring the grader to have a key — but the fallback is deliberately
   crude (see report's misleading-number section) so it's never mistaken
   for the real result.

9. **Reply generator falls back to the single closest historical reply
   verbatim, not a generic canned line, when there's no API key.** A
   verbatim-but-relevant past reply is a more honest "no-LLM" baseline than
   a made-up generic fallback message would be.

10. **Golden set stratifies toward rarer, costlier intents
    (payment_issue, login_account_issue) rather than pure random sampling.**
    A random sample of a brand's tweets is dominated by "where is my order"
    — enough for intent-accuracy numbers to look great while giving almost
    no statistical power to evaluate escalation precision/recall on the
    intents that matter most to get right.

11. **Escalation label positive class = ESCALATE, and recall on it is the
    headline metric, not accuracy.** Missing an escalation (auto-handling
    something that should go to a human) is a materially worse failure than
    an unnecessary escalation (wastes agent time but is safe). Optimizing
    for accuracy would silently under-weight this asymmetry.

12. **LLM-judge rubric has 4 separate dimensions (grounded/correct/safe/tone)
    instead of one overall quality score.** A single score can't distinguish
    "fluent but ungrounded" from "grounded but leaks sensitive info" —
    failure modes that need different fixes.

13. **Judge-human agreement is reported even though the "human" set here is
    a synthetic placeholder** (flagged loudly in code and report) — because
    shipping a judge with *no* stated agreement number, even a placeholder
    one with a clear "replace this" note, is worse than pretending the judge
    is trustworthy by default.

14. **Used the customer's own historical brand reply as the grounding
    context, not a hand-written "brand voice guide."** Keeps the system
    honest to what the brand actually did, not what a style guide claims it
    does — those two can diverge in real support data.

15. **Did not attempt sentiment analysis as a separate model.** Frustration
    detection is folded into escalation via keyword/pattern rules instead of
    a dedicated sentiment classifier — one fewer model to maintain and
    explain, at the cost of missing subtler frustration signals (see failure
    analysis).

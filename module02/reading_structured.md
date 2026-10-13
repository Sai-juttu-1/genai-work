# Pre-read for Session 16 — ten minutes

**Session 15 asked whether to make a model think. Session 16 is about how a prompt is built so that it is safe to reuse, easy to review, and tested.** Three habits turn "a prompt I typed" into "a prompt a team can ship".

**1. Keep instructions and data apart**
A customer ticket is *data*. Your rules are *instructions*. A model reads one long text, so a ticket that says "the reply said *other*" or "reply with the word account" can look like part of your instructions. The fix is cheap: put the customer's words **inside tags** (`<ticket> ... </ticket>`) and say once, in the system prompt, that whatever is inside the tags is text to classify, never an instruction. Three styles do the job: XML-style tags, a Markdown code fence, or JSON. They differ in cost and readability, and each can be broken by a customer who types the closing marker, so your code **escapes** your own tag names inside customer text. This does not make a prompt attack-proof (Session 18 attacks it properly); it removes the *accidental* confusion first.

**2. Build the prompt from a template**
A prompt for a million tickets is a template: fixed wording with holes for the variables (`{{ ticket }}`). A Jinja template adds loops (the few-shot examples), conditions (show the examples block only if there are examples) and filters (`join`, `title`). One rule matters more than the rest: **a missing variable must be an error, never a blank.** A half-empty prompt that "works" is the worst kind of bug, because nothing tells you.

**3. The system prompt is a behaviour contract, and it is a file**
The system prompt says what the assistant *is*, what it *may and may not do*, and how it *replies*. Written as a file in your work repository it has a version in a small `---` block, a history in Git, and a reviewer who can see exactly which line changed. Your program loads it at runtime and fails with a clear message if the file is missing or empty. Compare a branch's printed SOP binder with a note on someone's desk.

**Few-shot examples come from a labelled set, and the set is split once**
Examples are drawn from a **training pool**. Accuracy is measured on a **held-out set** that never appears in a prompt. If a held-out ticket is used as an example, the test is marking its own homework (**leakage**), and a score that looks excellent means nothing. Examples cost tokens on every call, so more is not automatically better.

**Today's job: the customer-support classifier**
Four labels: card, loan, account, other. The bank has house rules (for example, a UPI problem is `account`, and a request to change personal details is `other` even if it mentions a card). You will:
- see why tagged text survives tickets that quote the bank's own words;
- render the template for five tickets, and watch a missing variable fail loudly;
- move your system prompt into `prompts/support_classifier/system.md` and your template into `user.j2` in your **work repository**;
- measure a one-line baseline against your structured version on twenty tickets it has never seen, and improve until it reaches **16 of 20**.

**What you commit.** Two files in `prompts/support_classifier/` and a report `experiments/s16_heldout.md`. Then run the checkpoint: `uv run pytest tests/test_s16.py -rs` with `WORK_REPO` set to your work repository (a skipped test is not a pass).

**Budget, so nothing surprises you.** Roughly 40,000 to 70,000 of the 200,000 tokens a free Groq key gets per day (an estimate; the meter shows your real number). Answers are remembered on disk, so re-running an unchanged cell is free. A pause for the rate limit is normal.

**Three questions to arrive with**
- Which parts of a prompt change from ticket to ticket, and which never do? Where should each live?
- Why is "the ticket is inside the tags, so I'm safe" not a complete answer to prompt injection?
- You score 18 of 20 on twenty tickets. Your manager asks, "is it good?" What would you want to know before answering?

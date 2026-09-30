# Defend This Project — Q&A Sheet

Practice answering these out loud, pointing at the actual code (`app.py`) as
you go. This is exactly the "walk me through your project" + "explain
challenges" pattern that TCS, Cognizant, Accenture, Capgemini, and
LTIMindtree all reportedly ask.

---

**Q: Walk me through this project.**
It's a chatbot that answers questions about a company leave policy document.
I load the document, split it into chunks, embed the chunks and store them
in a FAISS vector database. When a user asks a question, I retrieve the most
relevant chunks and pass them to the LLM as context so it answers using the
actual policy text instead of guessing. I also added one tool — a date
calculator — because LLMs aren't reliable at precise date math, so the model
can call real Python code for that instead of trying to compute it itself.

**Q: Why did you chunk at 300 characters with 40 overlap?**
The policy sections are short, so smaller chunks keep each one focused on a
single topic (e.g. just sick leave), which improves retrieval precision. The
40-character overlap prevents a sentence from being split awkwardly across
two chunks and losing meaning at the boundary. In a longer/denser document
I'd likely increase chunk size and test retrieval quality empirically.

**Q: Why FAISS instead of Pinecone or another hosted vector DB?**
FAISS runs locally with no external service or cost, which was right for a
small, single-document demo project. For a production system with many
documents, frequent updates, or multiple users, I'd move to a hosted
option like Pinecone or Azure AI Search for scalability, persistence, and
easier managed indexing.

**Q: What happens if the retrieved chunks don't actually answer the question?**
Right now the system prompt instructs the model to answer only from the
provided context, which reduces (but doesn't eliminate) hallucination. A
real improvement would be explicitly instructing the model to say "I don't
have that information" when the retrieved context isn't relevant enough,
and possibly adding a relevance-score threshold on retrieval so weak matches
aren't passed to the LLM at all.

**Q: Why add a tool instead of just asking the LLM to count the days?**
LLMs generate text by predicting likely tokens, not by actually computing —
they're often wrong on multi-step arithmetic like counting weekdays across a
date range. Rather than trust that, I give the model a real Python function
it can call, so the actual counting is done by deterministic code, not a
guess dressed up as an answer.

**Q: Walk me through exactly what happens when the tool gets called.**
The LLM's response includes a `tool_calls` field instead of plain text —
that's it saying "call `calculate_business_days` with these arguments,"
as structured JSON. It does not run any code. My code reads that request,
calls the real `calculate_business_days()` function, gets the actual
integer result, and sends it back to the model in a second API call as a
`role: "tool"` message. Only then does the model generate the final
natural-language answer using that real result.

**Q: Why two LLM calls for a tool-using question, instead of one?**
The first call is where the model decides whether a tool is needed and with
what arguments. The second call happens after the tool has actually run, so
the model can incorporate the real result into a coherent final answer. The
model can't use a result it doesn't have yet — the two calls represent two
distinct steps in the reasoning loop.

**Q: How would this need to change to handle 100,000 documents instead of one?**
I'd move off FAISS to a hosted, scalable vector database, add metadata
filtering (e.g. by department or document type) so retrieval doesn't search
irrelevant categories, likely add a hybrid search (keyword + semantic) for
precision, and add a re-ranking step after initial retrieval. I'd also add
proper document versioning and access control, since not every user should
see every document.

**Q: How would you evaluate whether this system is working well?**
Check whether retrieved chunks are actually relevant to the question
(retrieval quality), separately from whether the final answer is accurate
and grounded in that context (generation quality). In practice: a small set
of test questions with known correct answers, checking for hallucination,
and tracking response latency and, eventually, real user feedback.

**Q: What's a security concern with a RAG system like this?**
If the source documents could be edited by untrusted users, someone could
embed hidden instructions in a document designed to manipulate the LLM's
behavior when that chunk gets retrieved — a prompt injection risk. Mitigations
include treating retrieved content strictly as data (not instructions) in
the prompt structure, and validating/sanitizing source documents before
ingestion.

**Q: What would you improve if you had more time?**
Add short-term memory so follow-up questions work in context (e.g. "and for
maternity leave?" after a prior question), add a second, more realistic tool
(like checking a mock employee's remaining leave balance), and deploy it
behind a simple FastAPI endpoint in Docker so it's actually integration-ready
rather than a local CLI script.

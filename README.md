# DocQA Assistant — RAG + Tool-Calling HR Policy Chatbot

A small chatbot that answers questions about a company leave policy document.
It combines **RAG** (to ground answers in the actual policy text) with
**tool/function calling** (to correctly do date math an LLM can't reliably do
on its own).

## Why this project exists

Built as a hands-on demonstration of core GenAI Developer skills:
document chunking, embeddings, vector search (FAISS), retrieval-augmented
generation, prompt design, and agentic tool calling — the core skills listed
in most GenAI Developer job descriptions.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-here"   # free at https://aistudio.google.com/app/apikey
python app.py
```

No credit card or billing setup needed — Google's Gemini API free tier works
with just a Google account.

## Example interaction

```
You: How many days of sick leave do I get?
Assistant: You are entitled to 12 days of paid sick leave per calendar year...

You: How many working days are there between 2026-09-21 and 2026-09-25?
Assistant: There are 5 working days between September 21 and September 25, 2026.
```

The first question is answered purely from retrieved context (RAG).
The second question triggers the `calculate_business_days` tool, because
counting weekdays precisely is not something an LLM should be trusted to do
by "reasoning" alone.

## How it works (architecture)

```
leave_policy.txt
      |
      v
  [chunking]  (RecursiveCharacterTextSplitter, 300 chars, 40 overlap)
      |
      v
  [embeddings]  (Gemini gemini-embedding-001)
      |
      v
  [FAISS vector store]
      |
      v
user question --> [similarity_search, k=3] --> retrieved chunks
      |
      v
[system prompt + retrieved context + question] --> Gemini (gemini-2.0-flash)
      |
      v
  +--> if answerable from context --> final answer
      |
      +--> if needs date math --> model requests calculate_business_days(...)
                 |
                 v
           app.py executes the real function
                 |
                 v
           result sent back to the model --> final answer
```

## Resume bullet (suggested)

> Built a RAG-based HR policy assistant using LangChain, FAISS, and the
> Google Gemini API; implemented document chunking, semantic retrieval, and a
> tool-calling agent capable of precise date calculations beyond the LLM's
> native reasoning.

## Possible extensions (mention these if asked "what would you improve")

- Add a second tool, e.g. checking a mock "remaining leave balance" per employee
- Swap FAISS for a hosted vector DB (Pinecone/Azure AI Search) for production scale
- Add a re-ranking step after retrieval for better precision
- Add short-term memory so follow-up questions ("and for maternity leave?") work in context
- Deploy behind a FastAPI endpoint + Docker for a real integration story

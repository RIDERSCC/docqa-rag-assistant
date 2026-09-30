"""
DocQA Assistant — A RAG + Tool-Calling chatbot over a company policy document.
Now using Google Gemini (free tier) instead of OpenAI, for both embeddings
and chat/tool-calling — no billing account required to run this.

What this project demonstrates (mapped to core GenAI concepts):
  1. Document loading & chunking
  2. Embeddings + a vector database (FAISS) for semantic retrieval
  3. RAG: retrieved context is inserted into the prompt so answers are grounded
  4. Tool/function calling: the LLM can call a real Python function (date math)
     when the question needs something an LLM can't reliably compute itself
  5. The agent loop: LLM decides -> app code executes -> result fed back -> final answer

Setup:
  1. Get a free API key at https://aistudio.google.com/app/apikey (no credit card needed)
  2. export GOOGLE_API_KEY="your-key-here"
  3. pip install -r requirements.txt
  4. python app.py
"""

import os
from datetime import datetime, timedelta

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from google import genai
from google.genai import types

DOC_PATH = "sample_docs/leave_policy.txt"
CHAT_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"


# ---------------------------------------------------------------------------
# STEP 1 & 2: Load the document and split it into chunks
# ---------------------------------------------------------------------------
def load_and_chunk(path: str):
    loader = TextLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_documents(documents)
    print(f"[setup] Loaded '{path}' and split it into {len(chunks)} chunks.")
    return chunks


# ---------------------------------------------------------------------------
# STEP 3: Embed the chunks and store them in a vector database (FAISS)
# ---------------------------------------------------------------------------
def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[setup] Embedded chunks (via Gemini) and built the FAISS vector store.")
    return vector_store


# ---------------------------------------------------------------------------
# STEP 4: Retrieve the most relevant chunks for a user's question
# ---------------------------------------------------------------------------
def retrieve_context(vector_store, query: str, k: int = 3) -> str:
    relevant_chunks = vector_store.similarity_search(query, k=k)
    return "\n\n".join(chunk.page_content for chunk in relevant_chunks)


# ---------------------------------------------------------------------------
# TOOL: a real function the LLM can call when it needs precise date math.
# LLMs are unreliable at counting weekdays across a date range, so this is a
# genuine example of "why would an agent need a tool" rather than a toy one.
# ---------------------------------------------------------------------------
def calculate_business_days(start_date: str, end_date: str) -> int:
    """Count weekdays (Mon-Fri) between two dates, inclusive. Dates as YYYY-MM-DD."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = 0
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0-4 = Monday-Friday
            days += 1
        current += timedelta(days=1)
    return days


# Tool schema the LLM sees, in Gemini's function-declaration format. The LLM
# only ever reads this description — it never executes the function itself.
CALCULATE_BUSINESS_DAYS_DECLARATION = {
    "name": "calculate_business_days",
    "description": (
        "Calculate the number of working days (Mon-Fri) between two dates, "
        "inclusive. Use this whenever the user asks how many leave/working "
        "days fall between two specific dates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
        },
        "required": ["start_date", "end_date"],
    },
}

TOOLS = types.Tool(function_declarations=[CALCULATE_BUSINESS_DAYS_DECLARATION])
TOOL_CONFIG = types.GenerateContentConfig(tools=[TOOLS])


# ---------------------------------------------------------------------------
# STEP 5: The agent loop — combine RAG context + tool calling
# ---------------------------------------------------------------------------
def answer_question(client: "genai.Client", vector_store, user_question: str) -> str:
    context = retrieve_context(vector_store, user_question)

    system_prompt = (
        "You are a helpful HR assistant. Answer using ONLY the policy context "
        "provided below. If the question requires counting working days "
        "between two dates, call the calculate_business_days tool instead of "
        "counting yourself.\n\nPolicy context:\n" + context
    )

    config = types.GenerateContentConfig(
        tools=[TOOLS],
        system_instruction=system_prompt,
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=user_question)])
    ]

    # First call: the model either answers directly, or asks to use a tool
    response = client.models.generate_content(
        model=CHAT_MODEL, contents=contents, config=config
    )

    part = response.candidates[0].content.parts[0]

    if part.function_call:
        # The model decided it needs the tool. My code executes it — the
        # model never runs Python itself, it only requested this call.
        function_call = part.function_call
        result = calculate_business_days(**function_call.args)

        # Add the model's function-call turn, then our function's real result,
        # back into the conversation.
        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": result},
                    )
                ],
            )
        )

        # Second call: give the tool's real result back so the model can
        # produce the final, natural-language answer.
        final_response = client.models.generate_content(
            model=CHAT_MODEL, contents=contents, config=config
        )
        return final_response.text

    return response.text


def main():
    client = genai.Client()  # reads GOOGLE_API_KEY from environment

    chunks = load_and_chunk(DOC_PATH)
    vector_store = build_vector_store(chunks)

    print("\nDocQA Assistant ready (running on Gemini free tier). Ask about the leave policy (or 'quit').\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        answer = answer_question(client, vector_store, question)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
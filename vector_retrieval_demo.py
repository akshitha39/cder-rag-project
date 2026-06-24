"""
vector_retrieval_demo.py
------------------------
System B: Vector RAG using ChromaDB and semantic embeddings.

Embeds CDER curriculum text, stores it in a ChromaDB vector database,
and retrieves semantically relevant chunks to ground LLM responses.
This reduces hallucination compared to the LLM-only baseline.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
)

# CDER curriculum texts — replace or extend with full chapter content
CDER_TEXTS = [
    "A thread is the fundamental unit of execution in parallel computing.",
    "A mutex is used to ensure mutual exclusion and prevent race conditions.",
    "A deadlock occurs when multiple threads wait on each other indefinitely.",
    "A race condition happens when two or more threads access shared data simultaneously "
    "and the outcome depends on the order of execution.",
    "A critical section is a segment of code that must not be executed by more than one "
    "thread at a time to prevent data corruption.",
    "Concurrency refers to multiple tasks making progress over overlapping time periods, "
    "but not necessarily at the same instant.",
    "Parallelism means multiple tasks execute simultaneously, typically on multiple CPU cores.",
    "Thread synchronization coordinates the execution of threads to ensure correct "
    "access to shared resources.",
    "A barrier in parallel programming is a synchronization point where all threads must "
    "wait until every thread has reached it before any can continue.",
    "Amdahl's Law states that the maximum speedup of a parallel program is limited by "
    "its sequential portion.",
]

# Load or create the ChromaDB vector store
vectordb = Chroma(
    persist_directory="./cder_vector_db",
    embedding_function=embeddings,
)

# Only embed texts if the database is empty (avoids duplicates on re-runs)
if len(vectordb.get()["ids"]) == 0:
    print("Embedding CDER texts into ChromaDB...")
    vectordb.add_texts(CDER_TEXTS)
    vectordb.persist()
    print("ChromaDB populated and persisted.")

# Initialize LLM and retriever
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=api_key,
)
retriever = vectordb.as_retriever()


def answer_question(query: str) -> str:
    """
    Answer a question using Vector RAG: retrieve relevant CDER text chunks,
    then generate a grounded response with the LLM.

    Args:
        query: The student's question about parallel computing.

    Returns:
        A grounded answer string based on the retrieved CDER context.
    """
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a teaching assistant for a Parallel Computing course.
Use ONLY the context below to answer the question. Do not add information not in the context.

Context:
{context}

Question: {query}

Answer in 2-3 sentences for a student."""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    sample_question = "What is a race condition?"
    print(f"Question: {sample_question}\n")
    answer = answer_question(sample_question)
    print(f"Vector RAG Answer:\n{answer}")

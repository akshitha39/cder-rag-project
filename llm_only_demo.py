"""
llm_only_demo.py
----------------
System A: LLM-Only baseline for the CDER GraphRAG project.

Sends questions directly to GPT-4o-mini with no retrieval grounding.
Used as a control to measure unguided model performance.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env file.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=api_key
)


def answer_llm(query: str) -> str:
    """
    Answer a question using only the LLM, with no retrieval context.

    Args:
        query: The question to answer.

    Returns:
        The LLM's response as a string.
    """
    response = llm.invoke(query)
    return response.content


if __name__ == "__main__":
    sample_question = "What is a race condition?"
    print(f"Question: {sample_question}\n")
    print("LLM Answer:")
    print(answer_llm(sample_question))

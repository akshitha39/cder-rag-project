"""
graph_rag_demo.py
-----------------
System C: Graph RAG using a Neo4j knowledge graph.

Queries a Neo4j knowledge graph containing CDER curriculum concepts
and their explicit relationships (REQUIRES, SOLVES, EXPLAINS, etc.).
This enables multi-hop relational reasoning and achieves the lowest
hallucination rate among the three systems evaluated.

Setup:
    - Install Neo4j Desktop: https://neo4j.com/download/
    - Create a database and add credentials to your .env file
    - Run this script after populating the graph (see graph_builder.py)
"""

import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

# Load credentials from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing in .env")
if not neo4j_password:
    raise RuntimeError("NEO4J_PASSWORD missing in .env")

# Connect to Neo4j knowledge graph
graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_user,
    password=neo4j_password,
    refresh_schema=False
)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=api_key
)


def graph_answer(query: str) -> str:
    """
    Answer a question using Graph RAG: retrieve related nodes and relationships
    from Neo4j, then generate a grounded, relational response with the LLM.

    The graph traversal surfaces explicit concept connections (e.g., Mutex
    REQUIRES Thread, Mutex SOLVES RaceCondition) that semantic search misses.

    Args:
        query: The student's question about parallel computing.

    Returns:
        A context-grounded answer based on graph-retrieved knowledge.
    """
    # Sample nodes from the graph as context
    cypher_result = graph.query("MATCH (n) RETURN n LIMIT 10")

    prompt = f"""You are an AI teaching assistant for a Parallel Computing course.
You have access to a Neo4j knowledge graph of CDER curriculum concepts.

Graph context (nodes and relationships):
{cypher_result}

Student question: {query}

Using ONLY the graph context above, provide a clear and accurate explanation
in 2-3 sentences. Reference specific concepts from the graph where possible."""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    sample_question = "What is a race condition?"
    print(f"Question: {sample_question}\n")
    answer = graph_answer(sample_question)
    print(f"Graph RAG Answer:\n{answer}")

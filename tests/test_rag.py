
import sys
import os
import asyncio

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from rag.retrieval import get_retriever

def test_rag_retrieval():
    print("Testing RAG Retrieval...")
    retriever = get_retriever()
    
    # query 1: General knowledge
    query = "What is the RSI interpretation?"
    print(f"\nQuery: {query}")
    results = retriever.retrieve(query)
    if results:
        print(f"Found {len(results)} results.")
        print(f"Top result: {results[0]['content'][:100]}...")
    else:
        print("No results found.")
        
    # query 2: Indian context
    query_india = "What are the market hours for NSE?"
    print(f"\nQuery: {query_india}")
    results_india = retriever.retrieve(query_india)
    if results_india:
        print(f"Found {len(results_india)} results.")
        print(f"Top result: {results_india[0]['content'][:100]}...")
    else:
        print("No results found.")
        
    if results and results_india:
        print("\nSUCCESS: RAG retrieval is working.")
    else:
        print("\nFAILURE: RAG retrieval failed.")

if __name__ == "__main__":
    test_rag_retrieval()

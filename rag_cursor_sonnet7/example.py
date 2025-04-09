import os
import asyncio
from dotenv import load_dotenv
from docling_lance_rag import DoclingLanceRAG

async def main():
    """
    Example demonstration of DoclingLanceRAG usage.
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize DoclingLanceRAG
    rag = DoclingLanceRAG(
        db_path="./lancedb_data",  # Path to LanceDB database
        # OpenAI API key will be loaded from environment variable
        embedding_model="text-embedding-ada-002",
        chat_model="gpt-4o",
        prefer_direct_text=True,
        ocr_enabled=True
    )
    
    # Example 1: Process documents in a directory
    docs_directory = "../docs"  # Change to your documents directory
    if os.path.exists(docs_directory):
        print(f"Processing documents in {docs_directory}...")
        await rag.add_documents(docs_directory)
    else:
        print(f"Directory {docs_directory} not found. Skipping document processing.")
    
    # Example 2: Answer a question using the RAG system
    question = "What is the key concept of document chunking in RAG systems?"
    print(f"\nQuestion: {question}")
    answer = await rag.answer_question(question, k=3)
    print(f"\nAnswer: {answer}")
    
    # Example 3: Retrieve and format context for external use (e.g., CrewAI)
    query = "document processing strategies"
    print(f"\nRetrieving context for: {query}")
    context = await rag.retrieve_and_format_context(query, k=2, transform_query=True)
    print(f"\nFormatted Context:\n{context}")
    
    # Example 4: Demonstrate query transformation
    original_query = "How do I extract text from PDF files?"
    print(f"\nOriginal Query: {original_query}")
    transformed_query = await rag._transform_query(original_query)
    print(f"Transformed Query: {transformed_query}")

if __name__ == "__main__":
    asyncio.run(main()) 
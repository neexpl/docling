import asyncio
from rag_cursor_auto.docling_lance_rag import DoclingLanceRAG
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Initialize the RAG pipeline with required parameters
    rag = DoclingLanceRAG(
        db_path="lancedb",
        openai_api_key=None,  # Will use OPENAI_API_KEY from environment
        embedding_model="text-embedding-ada-002",
        chat_model="gpt-4o-mini",
        table_name="docling_rag"
    )
    
    # Add documents
    await rag.add_documents("docs")
    
    # Example 1: Retrieve and format context
    context = await rag.retrieve_and_format_context(
        "What is the main topic of the documents?",
        k=3
    )
    print("Retrieved Context:")
    print(context)
    
    # Example 2: Direct question answering
    answer = await rag.answer_question(
        "What is the main topic of the documents?",
        k=3
    )
    print("\nGenerated Answer:")
    print(answer)

# Run the async code
asyncio.run(main())

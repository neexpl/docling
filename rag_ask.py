import asyncio
from rag_cursor_auto.docling_lance_rag import DoclingLanceRAG
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Initialize the RAG pipeline with the same parameters used to create the database
    rag = DoclingLanceRAG(
        db_path="lancedb",
        openai_api_key=None,  # Will use OPENAI_API_KEY from environment
        embedding_model="text-embedding-ada-002",
        chat_model="gpt-4o-mini",
        table_name="docling_rag"
    )
    
    while True:
        # Get user question
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        # Get number of context chunks to retrieve
        try:
            k = int(input("How many context chunks to retrieve? (default: 3): ") or "3")
        except ValueError:
            k = 3
            
        # Get answer
        answer = await rag.answer_question(question, k=k)
        
        # Print the answer
        print("\nAnswer:")
        print(answer)
        
        # Optionally show the context used
        show_context = input("\nShow context used? (y/n): ").lower() == 'y'
        if show_context:
            context = await rag.retrieve_and_format_context(question, k=k)
            print("\nContext used:")
            print(context)

# Run the async code
if __name__ == "__main__":
    asyncio.run(main()) 
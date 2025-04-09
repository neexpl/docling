# DoclingLanceRAG

A Python implementation of a Retrieval-Augmented Generation (RAG) pipeline using Docling for document processing, LanceDB for vector storage, and OpenAI for embeddings and text generation.

## Features

- **Asynchronous Operations**: All I/O-intensive operations are implemented asynchronously.
- **Document Processing**: Uses Docling for document loading, language detection, parsing, and chunking.
- **Vector Storage**: Uses LanceDB for efficient vector storage and similarity search.
- **Embeddings & LLM**: Uses OpenAI models for generating embeddings and text.
- **Query Transformation**: Transforms natural language questions into optimized search queries.
- **Full RAG Pipeline**: End-to-end question answering based on document context.
- **External Integration**: Provides methods for integrating with external tools like CrewAI.

## Installation

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install docling lancedb openai python-dotenv
```

3. Set up your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
import asyncio
from docling_lance_rag import DoclingLanceRAG

async def main():
    # Initialize DoclingLanceRAG
    rag = DoclingLanceRAG(
        db_path="./lancedb_data",  # Path to LanceDB database
        # OpenAI API key will be loaded from environment variable
        embedding_model="text-embedding-ada-002",
        chat_model="gpt-4o"
    )
    
    # Process documents
    await rag.add_documents("path/to/documents")
    
    # Answer a question
    answer = await rag.answer_question("What is RAG?")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

For more advanced usage, check the `example.py` file, which demonstrates:

1. Processing documents
2. Answering questions
3. Retrieving and formatting context for external use
4. Demonstrating query transformation

### Integration with CrewAI

To integrate with CrewAI, you can use the `retrieve_and_format_context` method to get formatted context that can be passed to a CrewAI agent:

```python
# In your CrewAI workflow
context = await rag.retrieve_and_format_context(
    query="Your search query",
    k=5,  # Number of chunks to retrieve
    transform_query=True  # Whether to transform the query
)

# Pass context to CrewAI agent
agent.task(f"Answer this question using the provided context: {context}")
```

## Configuration Options

- `db_path`: Path to the LanceDB database directory
- `openai_api_key`: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
- `embedding_model`: Name of the OpenAI embedding model to use
- `chat_model`: Name of the OpenAI chat model to use
- `prefer_direct_text`: Whether to prefer direct text extraction over OCR for PDFs
- `ocr_enabled`: Whether to enable OCR fallback for PDFs
- `table_name`: Name of the LanceDB table

## Class Methods

- `add_documents(directory_path)`: Process all documents in the specified directory
- `_process_file(file_path)`: Process a single document file (internal method)
- `_embed_and_store(chunks)`: Generate embeddings and store in LanceDB (internal method)
- `_transform_query(question)`: Transform a question into an optimized search query (internal method)
- `retrieve(query_text, k)`: Retrieve the top k most relevant chunks
- `retrieve_and_format_context(query_text, k, transform_query)`: Retrieve and format context for external use
- `answer_question(question, k)`: Answer a question using the RAG pipeline

## Document Processing Features

- Language detection for all documents
- PDF processing with direct text extraction
- Fallback to OCR with language-specific configuration when needed
- Hybrid chunking strategy for all documents
- Metadata extraction (source, language, page number, titles, descriptions)

## License

MIT 
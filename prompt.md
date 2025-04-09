# Prompt for AI Code Generation (e.g., feed this to an LLM)

"""
Generate asynchronous Python code for a class named `DoclingLanceRAG` that implements a full Retrieval-Augmented Generation (RAG) pipeline, including query transformation and final answer generation, while also providing components suitable for integration into a CrewAI workflow.

**Core Requirements:**

1.  **Asynchronous Operations:** All methods involving I/O (file reading, API calls, database operations) should be implemented using `async def` and `await`.
2.  **Document Processing (Docling):**
    * Use the `docling` library for loading and processing documents found within a specified directory.
    * Check if processed document is already in database.
    * Implement language detection using `Docling`.
    * For **PDF documents**:
        * Configure `Docling`'s PDF processing to **prioritize direct text extraction**.
        * **Only if a PDF requires OCR** (e.g., it is image-based or direct text extraction is insufficient/fails), then utilize Tesseract OCR.
        * When OCR is used, ensure it's configured with Tesseract and **pass the previously detected language code** to Tesseract's language parameter (e.g., `ocr_lang`) for improved accuracy.
    * Apply `Docling`'s **Hybrid Chunking** strategy to all processed documents.
    * Extract relevant metadata during processing: source filename, detected language, page number, document title, chapter title, and descriptions for any pictures, tables, or charts identified by Docling within the chunk's context. Handle cases where metadata might be missing gracefully.
3.  **Vector Storage (LanceDB):**
    * Use a local `LanceDB` vector database.
    * Initialize the database connection asynchronously if possible (or handle synchronously within async methods if the library requires it).
    * Define and use a LanceDB table schema including fields for: `vector`, `text`, `language`, `source_document`, `document_title`, `chapter_title`, `page_number`, `picture_description`, `table_description`, `chart_description`.
    * Store the processed text chunks and their corresponding vector embeddings along with all extracted metadata in the LanceDB table.
4.  **Embeddings & LLM (OpenAI):**
    * Use an OpenAI embedding model (e.g., 'text-embedding-ada-002' or newer) for generating embeddings.
    * Use an OpenAI chat model (e.g., 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo') for query transformation and final answer generation.
    * Utilize the asynchronous `openai` client (`AsyncOpenAI`) for all OpenAI API calls (embeddings and chat completions).
    * Manage OpenAI API key securely (e.g., expect it via environment variable or passed during initialization).
5.  **Query Transformation:**
    * Implement an internal asynchronous method (`_transform_query`) that takes a user's natural language question and uses the specified OpenAI chat model to generate a more effective search query (or hypothetical document snippet) optimized for vector retrieval.
6.  **Retrieval:**
    * Implement an asynchronous method (`retrieve`) to take a query text (potentially transformed), generate its embedding using the OpenAI model, and query LanceDB to find the top `k` most relevant text chunks based on vector similarity.
    * The retrieval method should return the text content and associated metadata of the relevant chunks.
7.  **Context Formatting for External Use (e.g., CrewAI):**
    * Implement an asynchronous method (`retrieve_and_format_context`) that takes a user query, potentially transforms it (optional, consider if needed here or just use raw query), calls `retrieve`, and formats the results into a string or structured format suitable for injection into an external agent's prompt context (like CrewAI). This method does **not** call the final generation LLM.
8.  **End-to-End Question Answering:**
    * Implement a public asynchronous method (`answer_question`) that performs the full RAG loop:
        * Takes the original user question.
        * Calls `_transform_query` to refine the question into a search query.
        * Calls `retrieve` using the transformed query to get relevant context.
        * Constructs a prompt for the OpenAI chat model, including the original question and the retrieved context, instructing it to answer based *only* on the provided context.
        * Calls the OpenAI chat model to generate the final answer.
        * Returns the generated natural language answer.
9.  **CrewAI Integration Context:** While the class now includes end-to-end answering (`answer_question`), the `retrieve_and_format_context` method remains available for scenarios where only context retrieval is needed for external tools like CrewAI agents.

**Class Structure Outline:**

* `__init__(self, db_path: str, openai_api_key: str = None, embedding_model: str = "text-embedding-ada-002", chat_model: str = "gpt-4o", ...) -> None`:
    * Initializes `AsyncOpenAI` client (handling API key).
    * Sets embedding model and chat model names.
    * Initializes LanceDB connection to `db_path` and ensures table schema.
    * Initializes `Docling` configuration.
* `async add_documents(self, directory_path: str) -> None`:
    * Scans directory, calls `_process_file` for each.
* `async _process_file(self, file_path: str) -> None`:
    * Loads with `Docling`, detects language.
    * Handles PDF processing (direct text extraction preferred, fallback to Tesseract OCR with detected language if needed).
    * Applies Hybrid Chunking.
    * Extracts chunks and metadata.
    * Calls `_embed_and_store`.
* `async _embed_and_store(self, chunks: list[dict]) -> None`:
    * Generates embeddings via async OpenAI client.
    * Prepares data for LanceDB schema.
    * Adds data to LanceDB asynchronously (or wrapped).
* `async _transform_query(self, question: str) -> str`:
    * Creates a prompt instructing the chat model to rephrase the question into an effective search query.
    * Calls the async OpenAI chat completion endpoint.
    * Parses and returns the transformed query string.
* `async retrieve(self, query_text: str, k: int = 5) -> list[dict]`:
    * Generates embedding for `query_text`.
    * Queries LanceDB for top `k` neighbors.
    * Returns list of dicts with text and metadata.
* `async retrieve_and_format_context(self, query_text: str, k: int = 5, transform_query: bool = False) -> str`:
    * Optionally calls `_transform_query` if `transform_query` is True.
    * Calls `retrieve` with the appropriate query.
    * Formats results into a string suitable for external context injection.
* `async answer_question(self, question: str, k: int = 5) -> str`:
    * Calls `transformed_query = await self._transform_query(question)`.
    * Calls `context_chunks = await self.retrieve(transformed_query, k)`.
    * Formats `context_chunks` into a string representation.
    * Creates a prompt for the chat model including instructions, the original `question`, and the formatted `context_chunks`.
    * Calls the async OpenAI chat completion endpoint with this prompt.
    * Parses and returns the final answer string.

**Include:**
* Necessary async import statements (`asyncio`, `openai`, `lancedb`, `docling`, `os`, `glob`).
* Robust error handling (`try...except`).
* Python type hinting.
* Clear docstrings for class and all methods.
* Considerations for batching during embedding and storing.
* Mention API key security (environment variables recommended).
"""
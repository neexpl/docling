import os
import glob
import asyncio
from typing import List, Dict, Any, Optional, Union

import lancedb
from lancedb.pydantic import LanceModel, Vector
from openai import AsyncOpenAI
from docling import Document
from docling.parsers import PDFParser
from docling.chunkers import HybridChunker
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class ChunkSchema(LanceModel):
    """Schema for storing document chunks in LanceDB."""
    vector: Vector(1536) = Field(description="Embedding vector")
    text: str = Field(description="Text content of the chunk")
    language: str = Field(description="Detected language of the chunk")
    source_document: str = Field(description="Source filename")
    document_title: Optional[str] = Field(description="Document title")
    chapter_title: Optional[str] = Field(description="Chapter or section title")
    page_number: Optional[int] = Field(description="Page number")
    picture_description: Optional[str] = Field(description="Description of pictures in context")
    table_description: Optional[str] = Field(description="Description of tables in context")
    chart_description: Optional[str] = Field(description="Description of charts in context")


class DoclingLanceRAG:
    """
    A class implementing a full Retrieval-Augmented Generation (RAG) pipeline
    using Docling for document processing, LanceDB for vector storage, and OpenAI
    for embeddings and text generation.
    
    This class provides asynchronous methods for document processing, query transformation,
    retrieval, and answer generation, making it suitable for integration with CrewAI.
    """
    
    def __init__(
        self,
        db_path: str,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        chat_model: str = "gpt-4o",
        prefer_direct_text: bool = True,
        ocr_enabled: bool = True,
        table_name: str = "documents"
    ) -> None:
        """
        Initialize the DoclingLanceRAG with necessary configurations.
        
        Args:
            db_path: Path to the LanceDB database directory
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            embedding_model: Name of the OpenAI embedding model to use
            chat_model: Name of the OpenAI chat model to use
            prefer_direct_text: Whether to prefer direct text extraction over OCR for PDFs
            ocr_enabled: Whether to enable OCR fallback for PDFs
            table_name: Name of the LanceDB table
        """
        # Load environment variables if API key not provided
        if not openai_api_key:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        
        # LanceDB setup
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(db_path)
        
        # Create table if it doesn't exist
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(
                self.table_name,
                schema=ChunkSchema,
                mode="create"
            )
        else:
            self.table = self.db.open_table(self.table_name)
            
        # Docling configuration
        self.chunker = HybridChunker()
        self.pdf_parser = PDFParser(
            prefer_direct_text=prefer_direct_text,
            ocr_enabled=ocr_enabled
        )
        
    async def add_documents(self, directory_path: str) -> None:
        """
        Process all documents in the specified directory and add them to the vector database.
        
        Args:
            directory_path: Path to the directory containing documents
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
            
        # Get all files in the directory
        file_paths = []
        for ext in ["*.pdf", "*.txt", "*.md", "*.html"]:
            file_paths.extend(glob.glob(os.path.join(directory_path, ext)))
            
        # Process files in parallel
        tasks = [self._process_file(file_path) for file_path in file_paths]
        await asyncio.gather(*tasks)
        
    async def _process_file(self, file_path: str) -> None:
        """
        Process a single document file, extract chunks, and store them in the database.
        
        Args:
            file_path: Path to the document file
        """
        try:
            # Load document with Docling
            document = Document(file_path)
            
            # Detect language
            language = document.detect_language()
            
            # Configure OCR language for PDF if OCR is needed
            if file_path.lower().endswith('.pdf'):
                self.pdf_parser.ocr_lang = language
                document.parse(parser=self.pdf_parser)
            else:
                document.parse()
                
            # Apply chunking
            document.chunk(chunker=self.chunker)
            
            # Extract chunks and metadata
            chunks_data = []
            for idx, chunk in enumerate(document.chunks):
                # Extract metadata
                metadata = chunk.metadata or {}
                source_document = os.path.basename(file_path)
                document_title = metadata.get('document_title', '')
                chapter_title = metadata.get('chapter_title', '')
                page_number = metadata.get('page_number')
                picture_description = metadata.get('picture_description', '')
                table_description = metadata.get('table_description', '')
                chart_description = metadata.get('chart_description', '')
                
                # Create chunk data dictionary
                chunk_data = {
                    'text': chunk.text,
                    'language': language,
                    'source_document': source_document,
                    'document_title': document_title,
                    'chapter_title': chapter_title,
                    'page_number': page_number,
                    'picture_description': picture_description,
                    'table_description': table_description,
                    'chart_description': chart_description
                }
                chunks_data.append(chunk_data)
                
            # Embed and store chunks
            await self._embed_and_store(chunks_data)
            
            print(f"Successfully processed and stored: {file_path}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    async def _embed_and_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Generate embeddings for the chunks and store them in LanceDB.
        
        Args:
            chunks: List of chunk data dictionaries
        """
        if not chunks:
            return
            
        # Extract text from chunks for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        BATCH_SIZE = 100
        all_embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        # Prepare data for LanceDB
        lance_data = []
        for chunk, embedding in zip(chunks, all_embeddings):
            lance_record = {
                'vector': embedding,
                'text': chunk['text'],
                'language': chunk['language'],
                'source_document': chunk['source_document'],
                'document_title': chunk['document_title'],
                'chapter_title': chunk['chapter_title'],
                'page_number': chunk['page_number'],
                'picture_description': chunk['picture_description'],
                'table_description': chunk['table_description'],
                'chart_description': chunk['chart_description']
            }
            lance_data.append(lance_record)
            
        # Add to LanceDB table
        # Note: LanceDB operations are synchronous, so we wrap it
        await asyncio.to_thread(self.table.add, lance_data)
        
    async def _transform_query(self, question: str) -> str:
        """
        Transform a natural language question into an optimized query for vector search.
        
        Args:
            question: The original user question
            
        Returns:
            str: The transformed query optimized for vector search
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a query optimization assistant. Your task is to rewrite the user's question "
                    "into a more effective search query for retrieving relevant information from a vector database. "
                    "Focus on extracting key concepts, expanding relevant terms, and removing unnecessary words. "
                    "Output only the transformed query with no additional explanation."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=prompt,
            temperature=0.0
        )
        
        transformed_query = response.choices[0].message.content.strip()
        return transformed_query
        
    async def retrieve(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top k most relevant chunks for the given query.
        
        Args:
            query_text: The query text to search for
            k: Number of chunks to retrieve
            
        Returns:
            List of dictionaries containing text content and metadata
        """
        # Generate embedding for the query
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=[query_text]
        )
        query_embedding = response.data[0].embedding
        
        # Search LanceDB for similar chunks
        # LanceDB search is synchronous, so we wrap it in to_thread
        search_results = await asyncio.to_thread(
            lambda: self.table.search(query_embedding).limit(k).to_list()
        )
        
        # Format results
        results = []
        for item in search_results:
            result = {
                'text': item['text'],
                'language': item['language'],
                'source_document': item['source_document'],
                'document_title': item['document_title'] if 'document_title' in item else None,
                'chapter_title': item['chapter_title'] if 'chapter_title' in item else None,
                'page_number': item['page_number'] if 'page_number' in item else None,
                'picture_description': item['picture_description'] if 'picture_description' in item else None,
                'table_description': item['table_description'] if 'table_description' in item else None,
                'chart_description': item['chart_description'] if 'chart_description' in item else None
            }
            results.append(result)
            
        return results
        
    async def retrieve_and_format_context(
        self, 
        query_text: str, 
        k: int = 5, 
        transform_query: bool = False
    ) -> str:
        """
        Retrieve context for a query and format it for use in external tools like CrewAI.
        
        Args:
            query_text: The query text or question
            k: Number of chunks to retrieve
            transform_query: Whether to transform the query before retrieval
            
        Returns:
            str: Formatted context string
        """
        if transform_query:
            search_query = await self._transform_query(query_text)
        else:
            search_query = query_text
            
        chunks = await self.retrieve(search_query, k)
        
        # Format context
        formatted_context = ""
        for i, chunk in enumerate(chunks):
            formatted_context += f"\n--- DOCUMENT CHUNK {i+1} ---\n"
            formatted_context += f"Text: {chunk['text']}\n"
            
            # Add metadata if available
            if chunk['source_document']:
                formatted_context += f"Source: {chunk['source_document']}\n"
            if chunk['document_title']:
                formatted_context += f"Document: {chunk['document_title']}\n"
            if chunk['chapter_title']:
                formatted_context += f"Section: {chunk['chapter_title']}\n"
            if chunk['page_number'] is not None:
                formatted_context += f"Page: {chunk['page_number']}\n"
            if chunk['picture_description']:
                formatted_context += f"Images: {chunk['picture_description']}\n"
            if chunk['table_description']:
                formatted_context += f"Tables: {chunk['table_description']}\n"
            if chunk['chart_description']:
                formatted_context += f"Charts: {chunk['chart_description']}\n"
                
        return formatted_context
        
    async def answer_question(
        self, 
        question: str, 
        k: int = 5
    ) -> str:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: The user's question
            k: Number of chunks to retrieve
            
        Returns:
            str: Generated answer based on retrieved context
        """
        # Transform the query
        transformed_query = await self._transform_query(question)
        
        # Retrieve relevant chunks
        context_chunks = await self.retrieve(transformed_query, k)
        
        # Format context for prompt
        formatted_context = ""
        for i, chunk in enumerate(context_chunks):
            formatted_context += f"\n--- DOCUMENT CHUNK {i+1} ---\n"
            formatted_context += f"Text: {chunk['text']}\n"
            
            # Add metadata if available
            if chunk['source_document']:
                formatted_context += f"Source: {chunk['source_document']}\n"
            if chunk['document_title']:
                formatted_context += f"Document: {chunk['document_title']}\n"
            if chunk['chapter_title']:
                formatted_context += f"Section: {chunk['chapter_title']}\n"
            if chunk['page_number'] is not None:
                formatted_context += f"Page: {chunk['page_number']}\n"
            if chunk['picture_description']:
                formatted_context += f"Images: {chunk['picture_description']}\n"
            if chunk['table_description']:
                formatted_context += f"Tables: {chunk['table_description']}\n"
            if chunk['chart_description']:
                formatted_context += f"Charts: {chunk['chart_description']}\n"
        
        # Create prompt for answer generation
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based only on the provided context. "
                    "If the answer cannot be found in the context, say 'I don't have enough information to answer that question.' "
                    "Do not make up information or use your own knowledge to answer the question."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context:\n{formatted_context}\n\n"
                    "Please answer the question based only on the provided context."
                )
            }
        ]
        
        # Generate answer
        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=prompt,
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip() 
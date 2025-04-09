import asyncio
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import glob

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from openai import AsyncOpenAI
import lancedb
import numpy as np
from pydantic import BaseModel
import pyarrow as pa

class ChunkMetadata(BaseModel):
    """Schema for document chunk metadata."""
    source_document: str
    language: str
    page_number: Optional[int]
    document_title: Optional[str]
    chapter_title: Optional[str]
    picture_description: Optional[str]
    table_description: Optional[str]
    chart_description: Optional[str]

class DoclingLanceRAG:
    """A class implementing RAG pipeline components using Docling for document processing,
    LanceDB for vector storage, and OpenAI for embeddings.
    
    This class is designed to be integrated into CrewAI workflows, providing document
    processing, storage, and retrieval capabilities while leaving the final LLM generation
    to the CrewAI agent.
    """
    
    def __init__(
        self,
        db_path: str,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        chat_model: str = "gpt-4o-mini",
        table_name: str = "docling_rag"
    ) -> None:
        """Initialize the RAG pipeline components.
        
        Args:
            db_path: Path to the LanceDB database directory
            openai_api_key: OpenAI API key (if None, will try to get from environment)
            embedding_model: Name of the OpenAI embedding model to use
            chat_model: Name of the OpenAI chat model to use for query transformation and answer generation
            table_name: Name of the LanceDB table to use
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.table_name = table_name
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        if not self.openai_client.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        # Initialize LanceDB
        self.db = lancedb.connect(db_path)
        self._ensure_table_exists()
        
        # Initialize Docling DocumentConverter
        pdf_options = PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_ocr=False,  # Don't do OCR by default
                ocr_options=TesseractCliOcrOptions(lang=["auto"]),
                backend="pypdf"
            )
        )
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_options}
        )
        
        # Create a table to track processed documents
        self._ensure_processed_docs_table_exists()
    
    def _ensure_table_exists(self) -> None:
        """Ensure the LanceDB table exists with the correct schema."""
        schema = pa.schema([
            ("vector", pa.list_(pa.float32(), 1536)),  # OpenAI embedding dimension
            ("text", pa.string()),
            ("source_document", pa.string()),
            ("language", pa.string()),
            ("page_number", pa.int32()),
            ("document_title", pa.string()),
            ("chapter_title", pa.string()),
            ("picture_description", pa.string()),
            ("table_description", pa.string()),
            ("chart_description", pa.string())
        ])
        
        if self.table_name not in self.db.table_names():
            self.db.create_table(self.table_name, schema=schema)
        self.table = self.db.open_table(self.table_name)
    
    def _ensure_processed_docs_table_exists(self) -> None:
        """Ensure the processed documents tracking table exists."""
        processed_docs_schema = pa.schema([
            ("file_path", pa.string()),
            ("last_modified", pa.int64()),  # Store as milliseconds since epoch
            ("processed_at", pa.int64()),   # Store as milliseconds since epoch
            ("status", pa.string()),
            ("error", pa.string())
        ])
        
        self.processed_docs_table_name = f"{self.table_name}_processed_docs"
        if self.processed_docs_table_name not in self.db.table_names():
            self.db.create_table(self.processed_docs_table_name, schema=processed_docs_schema)
        self.processed_docs_table = self.db.open_table(self.processed_docs_table_name)
    
    async def _is_document_processed(self, file_path: str) -> bool:
        """Check if a document has already been processed and is up to date.
        
        Args:
            file_path: Path to the document to check
            
        Returns:
            True if the document is already processed and up to date, False otherwise
        """
        try:
            # Get file's last modification time in milliseconds
            last_modified = int(os.path.getmtime(file_path) * 1000)
            
            # Check if document exists in processed docs table
            results = await asyncio.to_thread(
                lambda: self.processed_docs_table.to_pandas()
            )
            
            # Filter for the specific file path
            doc_entry = results[results['file_path'] == file_path]
            
            if doc_entry.empty:
                return False
                
            # Check if file has been modified since last processing
            processed_doc = doc_entry.iloc[0]
            return processed_doc['last_modified'] == last_modified and processed_doc['status'] == 'success'
            
        except Exception as e:
            print(f"Error checking document status for {file_path}: {str(e)}")
            return False
    
    async def _mark_document_processed(self, file_path: str, status: str = "success", error: Optional[str] = None) -> None:
        """Mark a document as processed in the tracking table.
        
        Args:
            file_path: Path to the processed document
            status: Processing status ("success" or "error")
            error: Error message if status is "error"
        """
        try:
            # Convert timestamps to milliseconds
            last_modified = int(os.path.getmtime(file_path) * 1000)
            processed_at = int(asyncio.get_event_loop().time() * 1000)
            
            data = {
                "file_path": file_path,
                "last_modified": last_modified,
                "processed_at": processed_at,
                "status": status,
                "error": error or ""
            }
            
            # Remove any existing entry for this file
            await asyncio.to_thread(
                lambda: self.processed_docs_table.delete(f"file_path = '{file_path}'")
            )
            
            # Add new entry
            await asyncio.to_thread(self.processed_docs_table.add, [data])
            
        except Exception as e:
            print(f"Error marking document as processed: {str(e)}")
    
    async def add_documents(self, directory_path: str) -> None:
        """Process and store all documents in the specified directory.
        
        Args:
            directory_path: Path to directory containing documents to process
        """
        # Get all processable files
        file_patterns = ["*.pdf", "*.txt", "*.md", "*.docx"]
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(os.path.join(directory_path, pattern)))
        
        # Process files concurrently
        tasks = [self._process_file(file_path) for file_path in files]
        await asyncio.gather(*tasks)
    
    async def _process_file(self, file_path: str) -> None:
        """Process a single file using Docling and store its chunks.
        
        Args:
            file_path: Path to the file to process
        """
        try:
            # Check if document is already processed
            if await self._is_document_processed(file_path):
                print(f"Skipping already processed file: {file_path}")
                return
            
            # First attempt: Try direct text extraction
            result = self.converter.convert(file_path)
            doc = result.document
            
            # Check if text extraction was successful
            if not hasattr(doc, 'text') or not doc.text or len(doc.text.strip()) == 0:
                # If no text was extracted, enable OCR
                if file_path.lower().endswith('.pdf'):
                    pdf_options = self.converter.format_to_options[InputFormat.PDF]
                    pdf_options.pipeline_options.do_ocr = True
                    result = self.converter.convert(file_path)
                    doc = result.document
            
            # Get document content and metadata
            content = doc.export_to_markdown() if hasattr(doc, 'export_to_markdown') else ""
            metadata = doc.export_to_dict() if hasattr(doc, 'export_to_dict') else {}
            
            # If no content was extracted, try to get text directly
            if not content and hasattr(doc, 'pages'):
                content = "\n".join(page.text for page in doc.pages if hasattr(page, 'text'))
            
            # If still no content, try to get text from the document object
            if not content and hasattr(doc, 'text'):
                content = doc.text
            
            if not content:
                print(f"Warning: Could not extract text from {file_path}")
                await self._mark_document_processed(file_path, status="error", error="No text content extracted")
                return
            
            # Use Docling's hybrid chunking if available, otherwise create chunks manually
            if hasattr(doc, 'get_chunks'):
                chunks = doc.get_chunks(
                    strategy="hybrid",
                    chunk_size=1000,
                    overlap=200
                )
            else:
                # Manual chunking if get_chunks is not available
                chunks = []
                words = content.split()
                chunk_size = 1000
                overlap = 200
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_text = " ".join(words[i:i + chunk_size])
                    chunks.append(type('Chunk', (), {'text': chunk_text}))
            
            # Prepare chunks with metadata
            processed_chunks = []
            for chunk in chunks:
                chunk_metadata = ChunkMetadata(
                    source_document=file_path,
                    language=metadata.get('language', 'en'),
                    page_number=metadata.get('page_number'),
                    document_title=metadata.get('title'),
                    chapter_title=metadata.get('chapter_title'),
                    picture_description=metadata.get('picture_description'),
                    table_description=metadata.get('table_description'),
                    chart_description=metadata.get('chart_description')
                )
                processed_chunks.append({
                    'text': chunk.text if hasattr(chunk, 'text') else str(chunk),
                    'metadata': chunk_metadata
                })
            
            # Store chunks
            await self._embed_and_store(processed_chunks)
            
            # Mark document as successfully processed
            await self._mark_document_processed(file_path)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error processing file {file_path}: {error_msg}")
            await self._mark_document_processed(file_path, status="error", error=error_msg)
    
    async def _embed_and_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Generate embeddings and store chunks in LanceDB.
        
        Args:
            chunks: List of dictionaries containing 'text' and 'metadata'
        """
        # Batch process embeddings
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in batch]
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            embeddings = [data.embedding for data in response.data]
            
            # Prepare data for LanceDB
            data = []
            for chunk, embedding in zip(batch, embeddings):
                metadata = chunk['metadata']
                data.append({
                    'vector': embedding,
                    'text': chunk['text'],
                    'source_document': metadata.source_document,
                    'language': metadata.language,
                    'page_number': metadata.page_number,
                    'document_title': metadata.document_title,
                    'chapter_title': metadata.chapter_title,
                    'picture_description': metadata.picture_description,
                    'table_description': metadata.table_description,
                    'chart_description': metadata.chart_description
                })
            
            # Store in LanceDB
            await asyncio.to_thread(self.table.add, data)
    
    async def retrieve(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query_text: The query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing retrieved chunks and metadata
        """
        # Generate query embedding
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query_text
        )
        query_embedding = response.data[0].embedding
        
        # Search LanceDB
        results = await asyncio.to_thread(
            lambda: self.table.search(query_embedding).limit(k).to_list()
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result['text'],
                'metadata': ChunkMetadata(
                    source_document=result['source_document'],
                    language=result['language'],
                    page_number=result['page_number'],
                    document_title=result['document_title'],
                    chapter_title=result['chapter_title'],
                    picture_description=result['picture_description'],
                    table_description=result['table_description'],
                    chart_description=result['chart_description']
                )
            })
        
        return formatted_results
    
    async def retrieve_and_format_context(self, query_text: str, k: int = 5) -> str:
        """Retrieve and format context for CrewAI agent.
        
        Args:
            query_text: The query text to search for
            k: Number of results to return
            
        Returns:
            Formatted string containing retrieved context and metadata
        """
        results = await self.retrieve(query_text, k)
        
        formatted_context = "Retrieved Context:\n\n"
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            formatted_context += f"[Chunk {i}]\n"
            formatted_context += f"{result['text']}\n\n"
            formatted_context += f"Source: {metadata.source_document}"
            if metadata.page_number:
                formatted_context += f" | Page: {metadata.page_number}"
            if metadata.document_title:
                formatted_context += f" | Document: {metadata.document_title}"
            if metadata.chapter_title:
                formatted_context += f" | Chapter: {metadata.chapter_title}"
            formatted_context += "\n"
            
            # Add descriptions if available
            if metadata.picture_description:
                formatted_context += f"Picture: {metadata.picture_description}\n"
            if metadata.table_description:
                formatted_context += f"Table: {metadata.table_description}\n"
            if metadata.chart_description:
                formatted_context += f"Chart: {metadata.chart_description}\n"
            
            formatted_context += "\n"
        
        return formatted_context
    
    async def _transform_query(self, question: str) -> str:
        """Transform a user question into an effective search query.
        
        Args:
            question: The original user question
            
        Returns:
            A transformed query optimized for vector search
        """
        prompt = f"""Transform the following question into an effective search query that would help find relevant information in a document database. 
        Focus on key terms and concepts that would appear in relevant documents. Do not include any explanations or additional text.

        Original question: {question}

        Transformed query:"""
        
        response = await self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a query transformation assistant that helps optimize search queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    async def answer_question(self, question: str, k: int = 5) -> str:
        """Perform end-to-end question answering using RAG.
        
        Args:
            question: The user's question
            k: Number of context chunks to retrieve
            
        Returns:
            The generated answer
        """
        # Transform the question
        transformed_query = await self._transform_query(question)
        
        # Retrieve relevant context
        context_chunks = await self.retrieve(transformed_query, k)
        
        # Format context
        context_str = "Retrieved Context:\n\n"
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk['metadata']
            context_str += f"[Chunk {i}]\n"
            context_str += f"{chunk['text']}\n\n"
            context_str += f"Source: {metadata.source_document}"
            if metadata.page_number:
                context_str += f" | Page: {metadata.page_number}"
            if metadata.document_title:
                context_str += f" | Document: {metadata.document_title}"
            if metadata.chapter_title:
                context_str += f" | Chapter: {metadata.chapter_title}"
            context_str += "\n\n"
        
        # Create prompt for final answer generation
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

        Question: {question}

        Context:
        {context_str}

        Answer:"""
        
        # Generate final answer
        response = await self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip() 
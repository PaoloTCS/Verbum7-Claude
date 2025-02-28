# /Users/paolopignatelli/VerbumTechnologies/Verbum7-Claude/backend/app/core/semantic_processor.py
"""
app/core/semantic_processor.py
Processes semantic information from documents and domain names.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import openai
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticProcessor:
    """
    Processes semantic information from documents and domains.
    Uses OpenAI embeddings to compute semantic distances.
    """
    
    def __init__(self, upload_folder: str):
        """
        Initialize the semantic processor.
        
        Args:
            upload_folder: Path to uploaded files
        """
        self.upload_folder = upload_folder
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found in environment variables")
        
        # Set API key for legacy OpenAI package
        openai.api_key = self.openai_api_key
        
        # Try to import newer OpenAI client
        try:
            from openai import OpenAI
            self.newer_client = OpenAI(api_key=self.openai_api_key)
            self.use_newer_client = True
            logger.info("Using newer OpenAI client")
        except (ImportError, TypeError):
            self.use_newer_client = False
            logger.info("Using legacy OpenAI client")
            
        self.embeddings_cache = {}
    
    def compute_distances(self, items: List[Dict[str, Any]], level_id: Optional[str] = None) -> Dict[Tuple[str, str], float]:
        """
        Compute semantic distances between items.
        
        Args:
            items: List of items with name and optional description
            level_id: Optional ID of the current level for caching
            
        Returns:
            Dictionary mapping (item1_id, item2_id) to distance
        """
        try:
            distances = {}
            logger.info(f"Computing distances for {len(items)} items")
            
            # Get embeddings for each item
            item_embeddings = {}
            for item in items:
                embedding = None
                
                # If item has a document path, use that for embedding
                if 'documentPath' in item and item['documentPath']:
                    embedding = self._get_document_embedding(item['documentPath'])
                
                # Otherwise use the name and description
                if not embedding:
                    text = item['name']
                    if 'description' in item and item['description']:
                        text += ": " + item['description']
                    embedding = self._get_text_embedding(text)
                
                if embedding is not None:
                    item_embeddings[item['id']] = embedding
                else:
                    logger.warning(f"Could not generate embedding for item: {item['name']}")
                    # Create a random embedding as fallback to prevent crashes
                    item_embeddings[item['id']] = np.random.rand(1536)
            
            # Compute distances between all item pairs
            for i, item1_id in enumerate(item_embeddings.keys()):
                item1_embedding = item_embeddings[item1_id]
                for item2_id in list(item_embeddings.keys())[i+1:]:
                    item2_embedding = item_embeddings[item2_id]
                    distance = self._compute_distance(item1_embedding, item2_embedding)
                    distances[(item1_id, item2_id)] = distance
            
            return distances
            
        except Exception as e:
            logger.error(f"Error computing distances: {str(e)}")
            return {}
    
    def get_document_summary(self, document_path: str) -> str:
        """
        Get a summary of a document for display.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Summary of the document
        """
        try:
            # Extract text from PDF
            full_path = os.path.join(self.upload_folder, document_path)
            text = self._extract_text_from_pdf(full_path)
            
            if not text:
                return "Could not extract text from document"
            
            # Generate summary with OpenAI - handle different API versions
            try:
                if self.use_newer_client:
                    response = self.newer_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides concise document summaries."},
                            {"role": "user", "content": f"Please provide a short summary (maximum 200 words) of the following document:\n\n{text[:5000]}..."}
                        ],
                        temperature=0.7,
                        max_tokens=250
                    )
                    return response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides concise document summaries."},
                            {"role": "user", "content": f"Please provide a short summary (maximum 200 words) of the following document:\n\n{text[:5000]}..."}
                        ],
                        temperature=0.7,
                        max_tokens=250
                    )
                    return response["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                return f"AI summary unavailable. Document begins with: {text[:500]}..."
                
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            return f"Error summarizing document: {str(e)}"
    
    def process_document_query(self, document_path: str, query: str) -> str:
        """
        Process a query about a specific document.
        
        Args:
            document_path: Path to the document
            query: User query about the document
            
        Returns:
            Response to the query
        """
        try:
            if not self.openai_api_key:
                return "OpenAI API key not configured"

            # Extract text from PDF
            full_path = os.path.join(self.upload_folder, document_path)
            text = self._extract_text_from_pdf(full_path)
            
            if not text:
                return "Could not extract text from document"

            # Create OpenAI query with context - handle different API versions
            try:
                if self.use_newer_client:
                    response = self.newer_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant explaining concepts from documents."},
                            {"role": "user", "content": f"Based on this document content:\n\n{text[:4000]}...\n\nQuestion: {query}"}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    return response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant explaining concepts from documents."},
                            {"role": "user", "content": f"Based on this document content:\n\n{text[:4000]}...\n\nQuestion: {query}"}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    return response["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Error processing query with AI: {str(e)}")
                return f"Unable to process query with AI. Error: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error processing document query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self.embeddings_cache = {}
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            pdf = PdfReader(pdf_path)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def _get_document_embedding(self, document_path: str) -> Optional[np.ndarray]:
        """
        Get embedding for a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Document embedding vector
        """
        try:
            cache_key = f"doc:{document_path}"
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]
            
            # Extract text from PDF
            full_path = os.path.join(self.upload_folder, document_path)
            text = self._extract_text_from_pdf(full_path)
            
            if not text:
                return None
            
            # Generate summary for embedding
            summary = f"Document: {os.path.basename(document_path)}\n\nContent: {text[:2000]}"
            embedding = self._get_text_embedding(summary)
            
            if embedding is not None:
                self.embeddings_cache[cache_key] = embedding
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting document embedding: {str(e)}")
            return None
    
    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for text using OpenAI's API with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Cache handling remains the same
            cache_key = f"text:{hash(text)}"
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]
            
            embedding = None
            
            try:
                response = self.newer_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text[:8191]
                )
                embedding = np.array(response.data[0].embedding)
            except Exception as e:
                logger.error(f"OpenAI embedding error: {str(e)}")
                # Fall back to random embedding
                embedding = np.random.rand(1536)
            
            # Cache and return
            self.embeddings_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return np.random.rand(1536)
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute semantic distance between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Distance value (0-1, where 0 is identical)
        """
        try:
            # Compute cosine similarity and convert to distance
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return max(0, min(1, 1 - similarity))  # Ensure distance is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error computing distance: {str(e)}")
            return 0.5  # Return moderate distance on error
from typing import Dict, List, Optional, Tuple
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from src.config.config_loader import ConfigLoader
from src.querying.query_preprocessor import QueryPreprocessor
from src.querying.problem_decomposer import ProblemDecomposer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.prompting.prompt_manager import PromptManager
from src.utils.long_context_handler import LongContextHandler
from llm_providers import get_llm_provider
import logging
import traceback
from src.indexing.index_manager import IndexManager
from src.storing.storage_manager import StorageManager
from llama_index.core.indices.base import BaseIndex

class QueryEngine:
    """Manages the query processing pipeline."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.query_config = config_loader.get_query_config()
        self.index_manager = IndexManager(config_loader)
        self.storage_manager = StorageManager(config_loader)
        self.prompt_manager = PromptManager(config_loader)
        self.query_preprocessor = QueryPreprocessor(config_loader)
        self.context_handler = LongContextHandler(config_loader)
        self.problem_decomposer = ProblemDecomposer()
        
        # Load the index once during initialization if it exists
        self.index = self.storage_manager.load_index()
        if self.index:
            logging.info("Stored index loaded successfully.")
        else:
            logging.warning("No stored index found or failed to load.")
            
    def update_index(self, index: BaseIndex):
        """Update the index used by the query engine."""
        self.index = index
        logging.info("Query engine index updated.")

    def process_query(self, query: str, index: Optional[BaseIndex], provider_name: str,
                      model_name: str, force_rag: bool, prompt_type: str,
                      persona: Optional[str], max_output_tokens: int) -> Tuple[str, bool, Dict]:
        """Process a user query through the RAG pipeline."""
        try:
            logging.info(f"Processing query: {query} with provider: {provider_name}, model: {model_name}")
            
            # Preprocess the query
            processed_query, analysis = self.query_preprocessor.preprocess_query(query)
            analysis['original_query'] = query
            analysis['enhanced_query'] = processed_query
            
            # Decompose the query
            try:
                decomposition = self.problem_decomposer.decompose_query(processed_query)
                analysis['decomposition'] = decomposition
                # Log only sub-problem queries for brevity if available
                sub_queries = [sp.get('query', 'N/A') for sp in decomposition.get('sub_problems', [])]
                logging.info(f"Query decomposition successful: {sub_queries}")
            except Exception as e:
                logging.error(f"Error during query decomposition: {str(e)}")
                analysis['decomposition'] = {'error': str(e)} # Store error in analysis
            
            # Determine the index to use: session index > stored index
            active_index = index if index else self.index
            
            # Determine if RAG is needed
            analysis_needs_retrieval = self._check_if_needs_retrieval(processed_query, analysis)
            use_rag = force_rag or analysis_needs_retrieval
            
            logging.info(f"Force RAG: {force_rag}, Analysis needs retrieval: {analysis_needs_retrieval}, Use RAG: {use_rag}")

            if use_rag:
                if active_index:
                    logging.info("Attempting RAG query.")
                    # RAG Path
                    retrieval_config = self.query_config.get('retrieval', {})
                    hybrid_retriever = HybridRetriever(
                        active_index,
                        top_k=retrieval_config.get('rerank_top_k', 10),
                        weights=retrieval_config.get('hybrid_search_weights', {
                            'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2
                        })
                    )
                    
                    retrieved_nodes = hybrid_retriever.retrieve(processed_query)
                    
                    if retrieved_nodes:
                        logging.info(f"Retrieved {len(retrieved_nodes)} nodes.")
                        response = self.query_with_rag(
                            processed_query, retrieved_nodes, provider_name,
                            prompt_type, persona, max_output_tokens
                        )
                        analysis['rag_status'] = "Used RAG (Hybrid Retrieval)"
                    else:
                        logging.warning("RAG needed but no relevant documents found. Falling back to direct query.")
                        response = self.direct_query(processed_query, provider_name, prompt_type, persona)
                        analysis['rag_status'] = "RAG Attempted - No Docs Found (Fallback Direct)"
                else:
                    # RAG needed but no index available (neither session nor stored)
                    logging.warning("RAG needed but no index available. Falling back to direct query.")
                    response = self.direct_query(processed_query, provider_name, prompt_type, persona)
                    analysis['rag_status'] = "RAG Needed - No Index (Fallback Direct)"
            else:
                # Direct Query Path
                logging.info("Executing direct query.")
                response = self.direct_query(processed_query, provider_name, prompt_type, persona)
                analysis['rag_status'] = "Direct Query"
            
            return response, True, analysis
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
            return f"Error processing query: {str(e)}", False, {}
    
    def query_with_rag(self, query: str, retrieved_nodes: List, provider_name: str,
                      prompt_type: str, persona: Optional[str], max_output_tokens: int) -> str:
        """Process query with RAG using retrieved documents."""
        try:
            logging.debug(f"Processing RAG query: {query}")
            if not retrieved_nodes:
                logging.warning("query_with_rag called with no retrieved nodes.")
                return "No relevant context found to answer the query."
                
            # Extract text from retrieved nodes
            documents = [node.text for node in retrieved_nodes if hasattr(node, 'text') and node.text]
            if not documents:
                logging.error("Retrieved nodes have no valid text content.")
                return "Error: Retrieved context is empty or invalid."
            
            logging.debug(f"Extracted {len(documents)} text segments from nodes.")
            
            # Process documents using context handler
            if self.context_handler:
                context = self.context_handler.process_documents(
                    query=query,
                    documents=documents,
                    max_output_tokens=max_output_tokens
                )
                logging.debug(f"Processed context length: {len(context)} chars")
            else:
                context = "\n".join(documents)
                logging.warning("Context handler not available, using raw concatenated documents.")
            
            # Get response using prompt manager
            response = self.prompt_manager.get_response(
                query=query,
                context=context,
                provider_name=provider_name,
                prompt_type=prompt_type,
                persona=persona
            )
            logging.debug("Received response from prompt manager.")
            
            return response
            
        except Exception as e:
            logging.error(f"Error in RAG processing: {str(e)}\n{traceback.format_exc()}")
            return f"Error in RAG processing: {str(e)}"
    
    def direct_query(self, query: str, provider_name: str,
                    prompt_type: str, persona: Optional[str], max_output_tokens: int = 500) -> str: # Added max_output_tokens
        """Process query without retrieval."""
        try:
            logging.debug(f"Processing direct query: {query}")
            # Get response using prompt manager
            response = self.prompt_manager.get_response(
                query=query,
                context="", # No context for direct query
                provider_name=provider_name,
                prompt_type=prompt_type,
                persona=persona
            )
            logging.debug("Received response from prompt manager for direct query.")
            
            return response
            
        except Exception as e:
            logging.error(f"Error in direct query: {str(e)}\n{traceback.format_exc()}")
            return f"Error in direct query: {str(e)}"
    
    def _check_if_needs_retrieval(self, query: str, analysis: Dict) -> bool:
        """Determine if query needs retrieval based on analysis."""
        # Simplified check: if analysis contains certain keys indicating complexity or need for specific data
        retrieval_indicators = [
            'requires_calculation',
            'is_comparison',
            'time_sensitive',
            'requires_specific_info'
        ]
        
        needs_retrieval = any(analysis.get(key, False) for key in retrieval_indicators)
        
        # Also consider if companies are mentioned
        if len(analysis.get('companies_mentioned', [])) > 0:
            needs_retrieval = True
            
        logging.debug(f"Checking retrieval need for query: '{query}'. Analysis: {analysis}. Needs retrieval: {needs_retrieval}")
        return needs_retrieval 
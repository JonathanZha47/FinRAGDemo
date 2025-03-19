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

class QueryEngine:
    """Enhanced query engine with problem decomposition and hybrid retrieval."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.query_config = config_loader.get_query_config()
        if not self.query_config:
            raise ValueError("Query configuration is None. Please check config_loader.")
        self.query_preprocessor = QueryPreprocessor(config_loader)
        self.problem_decomposer = ProblemDecomposer()
        self.prompt_manager = PromptManager(config_loader)
        self.context_handler = None  # Will be initialized with model name when needed
        
    def process_query(
        self,
        query: str,
        index: Optional[object] = None,
        provider_name: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        force_rag: bool = False,
        prompt_type: str = "context",
        persona: Optional[str] = None,
        max_output_tokens: int = 1000
    ) -> Tuple[str, bool, Dict]:
        """Process a query and generate a response."""
        try:
            # Validate inputs
            if not query:
                return "Error: Query cannot be empty", False, {}
            
            # Initialize context handler with current model
            self.context_handler = LongContextHandler(model_name)
            
            # Preprocess the query
            try:
                processed_query, analysis = self.query_preprocessor.preprocess_query(query)
                if not processed_query:
                    return "Error: Query preprocessing failed", False, {}
            except Exception as e:
                logging.error(f"Error in query preprocessing: {str(e)}")
                return f"Error in query preprocessing: {str(e)}", False, {}
            
            # Decompose complex queries if enabled
            try:
                if self.query_config.get('decomposition', {}).get('enable_dependency_tracking', True):
                    decomposition = self.problem_decomposer.decompose_query(processed_query)
                    analysis['decomposition'] = decomposition
            except Exception as e:
                logging.error(f"Error in query decomposition: {str(e)}")
                # Continue processing even if decomposition fails
            
            # Determine if we need RAG
            try:
                needs_retrieval = force_rag or self._check_if_needs_retrieval(processed_query, analysis)
                if needs_retrieval and index:
                    # Validate retrieval configuration
                    retrieval_config = self.query_config.get('retrieval')
                    if not retrieval_config:
                        logging.warning("Retrieval configuration is missing, using defaults")
                        retrieval_config = {
                            'rerank_top_k': 10,
                            'hybrid_search_weights': {
                                'bm25': 0.4,
                                'vector': 0.4,
                                'keyword': 0.2
                            }
                        }
                    
                    # Use hybrid retrieval with configured weights
                    hybrid_retriever = HybridRetriever(
                        index,
                        top_k=retrieval_config.get('rerank_top_k', 10),
                        weights=retrieval_config.get('hybrid_search_weights', {
                            'bm25': 0.4,
                            'vector': 0.4,
                            'keyword': 0.2
                        })
                    )
                    
                    retrieved_nodes = hybrid_retriever.retrieve(processed_query)
                    if not retrieved_nodes:
                        logging.warning("No documents retrieved, falling back to direct query")
                        return self.direct_query(processed_query, provider_name, prompt_type, persona), True, {"rag_status": "No relevant documents found"}
                    
                    # Process with RAG
                    response = self.query_with_rag(
                        processed_query, retrieved_nodes, provider_name,
                        prompt_type, persona, max_output_tokens
                    )
                    analysis['rag_status'] = "Using hybrid retrieval for comprehensive results"
                else:
                    # Direct query without retrieval
                    response = self.direct_query(
                        processed_query, provider_name,
                        prompt_type, persona
                    )
                    analysis['rag_status'] = "Direct query"
                
                return response, True, analysis
                
            except Exception as e:
                logging.error(f"Error in retrieval process: {str(e)}\n{traceback.format_exc()}")
                return f"Error in retrieval process: {str(e)}", False, {}
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
            return f"Error processing query: {str(e)}", False, {}
    
    def query_with_rag(self, query: str, retrieved_nodes: List, provider_name: str,
                      prompt_type: str, persona: Optional[str], max_output_tokens: int) -> str:
        """Process query with RAG using enhanced retrieval."""
        try:
            if not retrieved_nodes:
                return "No relevant documents found"
                
            # Extract text from retrieved nodes
            documents = [node.text for node in retrieved_nodes if hasattr(node, 'text')]
            if not documents:
                return "Error: Retrieved nodes have no text content"
            
            # Process documents using context handler
            if self.context_handler:
                context = self.context_handler.process_documents(
                    query=query,
                    documents=documents,
                    max_output_tokens=max_output_tokens
                )
            else:
                context = "\n".join(documents)
            
            # Get response using prompt manager
            response = self.prompt_manager.get_response(
                query=query,
                context=context,
                provider_name=provider_name,
                prompt_type=prompt_type,
                persona=persona
            )
            
            return response
            
        except Exception as e:
            logging.error(f"Error in RAG processing: {str(e)}\n{traceback.format_exc()}")
            return f"Error in RAG processing: {str(e)}"
    
    def direct_query(self, query: str, provider_name: str,
                    prompt_type: str, persona: Optional[str]) -> str:
        """Process query without retrieval."""
        try:
            # Get response using prompt manager
            response = self.prompt_manager.get_response(
                query=query,
                context="",
                provider_name=provider_name,
                prompt_type=prompt_type,
                persona=persona
            )
            
            return response
            
        except Exception as e:
            logging.error(f"Error in direct query: {str(e)}\n{traceback.format_exc()}")
            return f"Error in direct query: {str(e)}"
    
    def _check_if_needs_retrieval(self, query: str, analysis: Dict) -> bool:
        """Determine if query needs retrieval based on analysis."""
        if not analysis:
            return False
            
        # Check various factors that might indicate retrieval is needed
        needs_retrieval = (
            analysis.get('requires_calculation', False) or
            analysis.get('is_comparison', False) or
            analysis.get('time_sensitive', False) or
            len(analysis.get('companies_mentioned', [])) > 0 or
            analysis.get('requires_specific_info', False)
        )
        
        return needs_retrieval 
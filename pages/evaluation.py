import streamlit as st
import pandas as pd
from src.evaluation.rag_evaluator import RAGEvaluator
from src.config.config_loader import ConfigLoader
from src.querying.query_engine import QueryEngine
from src.indexing.index_manager import IndexManager
from src.storing.storage_manager import StorageManager
from src.retrieval.hybrid_retriever import HybridRetriever
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

def display_metrics(metrics: dict):
    """Display evaluation metrics in a formatted way."""
    st.write("### Retrieval Metrics Overview")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hit Rate", f"{metrics['avg_hit_rate']:.3f}")
        st.metric("Precision", f"{metrics['avg_precision']:.3f}")
    
    with col2:
        st.metric("MRR", f"{metrics['avg_mrr']:.3f}")
        st.metric("Recall", f"{metrics['avg_recall']:.3f}")
    
    with col3:
        st.metric("NDCG", f"{metrics['avg_ndcg']:.3f}")
        st.metric("Avg Precision", f"{metrics['avg_ap']:.3f}")
    
    st.metric("Total Queries Evaluated", metrics['total_queries'])

def display_detailed_results(results_df: pd.DataFrame):
    """Display detailed evaluation results."""
    st.write("### Detailed Results")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Summary Statistics", "Query-Level Results"])
    
    with tab1:
        # Display summary statistics
        st.write("#### Metric Statistics")
        metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
        summary_stats = results_df[metrics].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Display correlation matrix
        st.write("#### Metric Correlations")
        correlation_matrix = results_df[metrics].corr()
        st.dataframe(correlation_matrix, use_container_width=True)
    
    with tab2:
        # Display per-query results with filtering
        st.write("#### Per-Query Results")
        metric_filter = st.selectbox(
            "Filter by Metric",
            metrics,
            format_func=lambda x: x.upper()
        )
        
        threshold = st.slider(
            f"Show queries with {metric_filter.upper()} above",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
        
        filtered_df = results_df[results_df[metric_filter] >= threshold]
        st.dataframe(filtered_df[metrics], use_container_width=True)

def app():
    st.title("RAG System Evaluation")
    st.write("Evaluate the performance of the retrieval system using various metrics.")
    
    # Initialize components
    config_loader = ConfigLoader()
    query_engine = QueryEngine(config_loader)
    index_manager = IndexManager(config_loader)
    storage_manager = StorageManager(config_loader)
    evaluator = RAGEvaluator(config_loader, query_engine)
    
    # Sidebar settings
    st.sidebar.subheader("Evaluation Settings")
    
    # QA Generation settings
    questions_per_chunk = st.sidebar.slider(
        "Questions per Document Chunk",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of questions to generate per document chunk"
    )
    
    # Retrieval settings
    st.sidebar.subheader("Retrieval Settings")
    top_k = st.sidebar.slider(
        "Top K Results",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of documents to retrieve"
    )
    
    # Hybrid search weights
    st.sidebar.write("Hybrid Search Weights")
    bm25_weight = st.sidebar.slider("BM25 Weight", 0.0, 1.0, 0.4, 0.1)
    vector_weight = st.sidebar.slider("Vector Weight", 0.0, 1.0, 0.4, 0.1)
    keyword_weight = st.sidebar.slider("Keyword Weight", 0.0, 1.0, 0.2, 0.1)
    
    # Normalize weights
    total_weight = bm25_weight + vector_weight + keyword_weight
    if total_weight > 0:
        weights = {
            'bm25': bm25_weight / total_weight,
            'vector': vector_weight / total_weight,
            'keyword': keyword_weight / total_weight
        }
    else:
        weights = {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2}
    
    # Main evaluation section
    if st.session_state.get('index') is None:
        st.warning("⚠️ Please upload and process documents in the main page first.")
        return
    
    # Add tabs for different evaluation stages
    generate_tab, evaluate_tab = st.tabs(["Generate QA Pairs", "Evaluate Retrieval"])
    
    with generate_tab:
        st.write("### Generate Question-Answer Pairs")
        if st.button("Generate New QA Pairs"):
            with st.spinner("Generating question-answer pairs..."):
                nodes = [node for node in st.session_state.index.docstore.docs.values()]
                qa_dataset = evaluator.generate_qa_pairs(nodes, questions_per_chunk)
                
                if qa_dataset:
                    st.success(f"✅ Successfully generated {len(qa_dataset.queries)} QA pairs!")
                    st.session_state.qa_dataset = qa_dataset
                else:
                    st.error("❌ Failed to generate QA pairs. Check the logs for details.")
    
    with evaluate_tab:
        st.write("### Evaluate Retrieval Performance")
        if st.button("Run Evaluation"):
            if not hasattr(st.session_state, 'qa_dataset'):
                try:
                    st.session_state.qa_dataset = EmbeddingQAFinetuneDataset.from_json("data/evaluation/qa_pairs.json")
                except Exception as e:
                    st.error("❌ No QA pairs found. Please generate them first.")
                    return
            
            with st.spinner("Evaluating retrieval performance..."):
                try:
                    # Load index
                    index = storage_manager.load_index()
                    if index is None:
                        st.error("❌ Failed to load index. Please check if the index exists.")
                        return
                    
                    # Initialize retriever with current settings
                    retriever = HybridRetriever(
                        index=index,
                        top_k=top_k,
                        weights=weights
                    )
                    
                    # Run evaluation
                    results_df, metrics = evaluator.evaluate_retriever(
                        retriever,
                        st.session_state.qa_dataset
                    )
                    
                    if not results_df.empty:
                        st.session_state.eval_results = {
                            'metrics': metrics,
                            'results_df': results_df
                        }
                        
                        # Display results
                        display_metrics(metrics)
                        
                        # Generate and display plots
                        st.write("### Visualization")
                        plots = evaluator.generate_evaluation_plots(results_df)
                        
                        plot_tab1, plot_tab2, plot_tab3 = st.tabs([
                            "Score Distribution",
                            "Metric Correlation",
                            "Performance Trend"
                        ])
                        
                        with plot_tab1:
                            st.plotly_chart(plots['score_distribution'], use_container_width=True)
                        with plot_tab2:
                            st.plotly_chart(plots['metric_correlation'], use_container_width=True)
                        with plot_tab3:
                            st.plotly_chart(plots['performance_trend'], use_container_width=True)
                        
                        # Display detailed results
                        display_detailed_results(results_df)
                    else:
                        st.error("❌ Evaluation failed. No results were generated.")
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    app() 
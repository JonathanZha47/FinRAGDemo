from typing import List, Dict, Tuple
import pandas as pd
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from src.querying.query_engine import QueryEngine
from src.config.config_loader import ConfigLoader
import plotly.graph_objects as go
import plotly.express as px
import os

class RAGEvaluator:
    """Evaluates RAG system performance using LLM-generated QA pairs."""
    
    def __init__(self, config_loader: ConfigLoader, query_engine: QueryEngine):
        self.config_loader = config_loader
        self.query_engine = query_engine
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.relevancy_evaluator = RelevancyEvaluator()
        
        # Define evaluation metrics
        self.metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
        
    def generate_qa_pairs(self, nodes: List, num_questions_per_chunk: int = 2) -> EmbeddingQAFinetuneDataset:
        """Generate QA pairs from document nodes using LLM."""
        try:
            # Create evaluation directory if it doesn't exist
            os.makedirs("data/evaluation", exist_ok=True)
            
            # Generate QA pairs
            qa_dataset = generate_question_context_pairs(
                nodes=nodes,
                num_questions_per_chunk=num_questions_per_chunk
            )
            
            # Save the dataset
            qa_dataset.save_json("data/evaluation/qa_pairs.json")
            print(f"Generated {len(qa_dataset.queries)} question-answer pairs")
            return qa_dataset
            
        except Exception as e:
            print(f"Error generating QA pairs: {str(e)}")
            return None
        
    def evaluate_retriever(self, retriever, qa_dataset: EmbeddingQAFinetuneDataset = None) -> Tuple[pd.DataFrame, Dict]:
        """Evaluate retriever performance using generated QA pairs."""
        try:
            if qa_dataset is None:
                # Load existing QA dataset
                qa_dataset = EmbeddingQAFinetuneDataset.from_json("data/evaluation/qa_pairs.json")
            
            # Initialize retriever evaluator
            retriever_evaluator = RetrieverEvaluator.from_metric_names(
                self.metrics,
                retriever=retriever
            )
            
            # Evaluate on entire dataset
            eval_results = retriever_evaluator.evaluate_dataset(qa_dataset)
            
            # Process results
            metric_dicts = []
            for eval_result in eval_results:
                metric_dicts.append(eval_result.metric_vals_dict)
            
            # Create detailed DataFrame
            results_df = pd.DataFrame(metric_dicts)
            
            # Calculate aggregate metrics
            metrics = {
                'avg_hit_rate': results_df['hit_rate'].mean(),
                'avg_mrr': results_df['mrr'].mean(),
                'avg_precision': results_df['precision'].mean(),
                'avg_recall': results_df['recall'].mean(),
                'avg_ap': results_df['ap'].mean(),
                'avg_ndcg': results_df['ndcg'].mean(),
                'total_queries': len(results_df)
            }
            
            return results_df, metrics
            
        except Exception as e:
            print(f"Error evaluating retriever: {str(e)}")
            return pd.DataFrame(), {}
    
    def generate_evaluation_plots(self, results_df: pd.DataFrame) -> Dict:
        """Generate visualization plots for evaluation results."""
        plots = {}
        
        try:
            # Score Distribution Plot
            fig_scores = go.Figure()
            for metric in self.metrics:
                fig_scores.add_trace(go.Box(y=results_df[metric], name=metric.upper()))
            fig_scores.update_layout(
                title='Distribution of Retrieval Metrics',
                yaxis_title='Score',
                showlegend=True
            )
            plots['score_distribution'] = fig_scores
            
            # Metric Correlation Plot
            fig_correlation = px.scatter_matrix(
                results_df[self.metrics],
                title='Correlation between Retrieval Metrics'
            )
            plots['metric_correlation'] = fig_correlation
            
            # Performance Trend
            fig_performance = go.Figure()
            for metric in self.metrics:
                fig_performance.add_trace(go.Scatter(
                    y=results_df[metric],
                    name=metric.upper(),
                    mode='lines+markers'
                ))
            fig_performance.update_layout(
                title='Performance Across Queries',
                xaxis_title='Query Index',
                yaxis_title='Score'
            )
            plots['performance_trend'] = fig_performance
            
            return plots
            
        except Exception as e:
            print(f"Error generating evaluation plots: {str(e)}")
            return {} 
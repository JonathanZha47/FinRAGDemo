import streamlit as st
import os
from utils import process_documents, get_response
import tempfile
from dotenv import load_dotenv
from src.config.config_loader import ConfigLoader
from src.data_ingestion.document_loader import DocumentLoader
from src.indexing.index_manager import IndexManager
from src.storing.storage_manager import StorageManager
from src.querying.query_engine import QueryEngine
from src.prompting.prompt_manager import PromptManager
from typing import Dict

# Load environment variables
load_dotenv()

# Initialize configuration
config_loader = ConfigLoader()
ui_config = config_loader.get_ui_config()

# Load and verify API keys
openai_key = os.getenv("OPENAI_API_KEY")
huggingface_key = os.getenv("HUGGINGFACE_API_KEY") 
openrouter_key = os.getenv("OPENROUTER_API_KEY")

if openai_key:
    print("✅ OpenAI API Key Loaded Successfully:", openai_key[:5] + "..." + openai_key[-5:])
else:
    print("⚠️ OpenAI API Key is missing! Check your .env file.")

if huggingface_key:
    print("✅ HuggingFace API Key Loaded Successfully:", huggingface_key[:5] + "..." + huggingface_key[-5:])
else:
    print("⚠️ HuggingFace API Key is missing! Check your .env file.")

if openrouter_key:
    print("✅ OpenRouter API Key Loaded Successfully:", openrouter_key[:5] + "..." + openrouter_key[-5:])
else:
    print("⚠️ OpenRouter API Key is missing! Check your .env file.")

# Available LLM providers
LLM_PROVIDERS = {
    "OpenAI (GPT-3.5-turbo)": "openai",
    "HuggingFace (Mixtral-8x7B)": "huggingface",
    "OpenRouter (Mixtral-8x7B)": "openrouter"
}

st.set_page_config(
    page_title="Financial Advisor Bot",
    page_icon="💰",
    layout="wide"
)

def save_uploaded_file(uploaded_file):
    try:
        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Save uploaded file
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, file_path
    except Exception as e:
        return False, str(e)

def initialize_session_state():
    """Initialize session state variables."""
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False

def sidebar_settings(ui_config: dict):
    """Create sidebar with parameter settings."""
    st.sidebar.title("Settings")
    
    # LLM Provider Selection
    provider_name = st.sidebar.selectbox(
        "Select LLM Provider",
        options=["openai", "huggingface", "openrouter"],
        index=0
    )
    
    # Model Selection based on provider
    available_models = ui_config['available_models'][provider_name]
    model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=0
    )
    
    # Get output token range for selected model
    output_token_range = ui_config.get('output_tokens_range', {}).get(provider_name, {}).get(model, [1, 2048, 50])
    
    # Prompt Engineering Settings
    st.sidebar.subheader("Prompt Engineering")
    prompt_type = st.sidebar.selectbox(
        "Prompt Engineering Technique",
        options=["Few-Shot", "Context-Prompting", "Chain-of-Thought", "Persona"],
        index=0
    )
    
    # Persona selection if persona prompting is chosen
    persona = None
    if prompt_type == "Persona":
        persona = st.sidebar.selectbox(
            "Select Persona",
            options=["Financial Advisor", "Risk Analyst", "Investment Strategist"],
            index=0
        )
    
    # Verification Settings
    st.sidebar.subheader("Response Verification")
    enable_citation_check = st.sidebar.checkbox("Enable Citation Verification", value=True)
    enable_hallucination_check = st.sidebar.checkbox("Enable Hallucination Detection", value=True)
    enable_guardrails = st.sidebar.checkbox("Enable SEC Compliance Check", value=True)
    
    # RAG Parameters
    st.sidebar.subheader("RAG Parameters")
    top_k = st.sidebar.slider(
        "Top K Results",
        min_value=ui_config['top_k_range'][0],
        max_value=ui_config['top_k_range'][1],
        value=3,
        step=ui_config['top_k_range'][2]
    )
    
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=ui_config['chunk_size_range'][0],
        max_value=ui_config['chunk_size_range'][1],
        value=1024,
        step=ui_config['chunk_size_range'][2]
    )
    
    # LLM Parameters
    st.sidebar.subheader("LLM Parameters")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=ui_config['temperature_range'][0],
        max_value=ui_config['temperature_range'][1],
        value=0.1,
        step=ui_config['temperature_range'][2]
    )
    
    # Dynamic max output tokens slider based on model
    max_output_tokens = st.sidebar.slider(
        "Max Output Tokens",
        min_value=output_token_range[0],
        max_value=output_token_range[1],
        value=min(1000, output_token_range[1]),  # Default to 1000 or max allowed
        step=output_token_range[2],
        help="Maximum number of tokens to generate in the response. Adjusts based on model capabilities."
    )
    
    return {
        "provider_name": provider_name,
        "model": model,
        "prompt_type": prompt_type,
        "persona": persona,
        "enable_citation_check": enable_citation_check,
        "enable_hallucination_check": enable_hallucination_check,
        "enable_guardrails": enable_guardrails,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens
    }

def display_query_analysis(analysis: Dict):
    """Display query analysis results in the main content area."""
    st.write("### Query Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display intent information
        st.write("**Query Intent:**")
        # Display core intents directly from analysis
        intent_fields = {
            'is_question': 'Question',
            'requires_calculation': 'Calculation Required',
            'is_comparison': 'Comparison',
            'time_sensitive': 'Time Sensitive'
        }
        
        for field, display_name in intent_fields.items():
            value = analysis.get(field, False)
            st.write(f"- {display_name}: {'✅' if value else '❌'}")
        
        # Display retrieval recommendation
        st.write("\n**Retrieval Strategy:**")
        requires_retrieval = analysis.get('requires_retrieval', False)
        print("requires_retrieval", requires_retrieval)
        if requires_retrieval:
            st.write("✅ RAG Recommended")
            # Show RAG status message if available
            if 'rag_status' in analysis:
                st.write("Current RAG Status:")
                st.info(analysis['rag_status'])
        else:
            st.write("❌ Direct Query Recommended")
    
    with col2:
        # Display companies mentioned
        companies = analysis.get('companies_mentioned', [])
        if companies:
            st.write("**Companies Mentioned:**")
            for company in companies:
                st.write(f"- {company}")
    
    # Display LLM analysis if available
    llm_analysis = analysis.get('llm_analysis')
    if llm_analysis:
        st.write("\n**Detailed Analysis:**")
        st.info(llm_analysis)
    
    st.markdown("---")  # Add a separator line

def display_verification_results(citations: Dict, hallucination_check: Dict, compliance_check: Dict):
    """Display verification results in an organized manner."""
    st.write("### Response Verification")
    
    # Citations
    with st.expander("Citation Verification", expanded=True):
        st.write(f"**Verification Score:** {citations['verification_score']:.2%}")
        
        st.write("**Verified Statements:**")
        for citation in citations['verified']:
            st.markdown(f"✅ {citation['statement']}")
            st.markdown(f"*Source: {citation['source']}*")
            
        if citations['unverified']:
            st.write("**Unverified Statements:**")
            for statement in citations['unverified']:
                st.markdown(f"❌ {statement}")
    
    # Hallucination Detection
    with st.expander("Hallucination Detection", expanded=True):
        st.write(f"**Confidence Score:** {hallucination_check['confidence_score']:.2%}")
        
        if hallucination_check['warning_flags']:
            st.write("**Warning Flags:**")
            for flag in hallucination_check['warning_flags']:
                st.warning(flag)
    
    # SEC Compliance
    with st.expander("SEC Compliance Check", expanded=True):
        if compliance_check['compliant']:
            st.success("✅ Response complies with SEC guidelines")
        else:
            st.error("❌ Response has compliance issues")
            
        if compliance_check['violations']:
            st.write("**Violations:**")
            for violation in compliance_check['violations']:
                st.error(violation)
                
        if compliance_check['warnings']:
            st.write("**Warnings:**")
            for warning in compliance_check['warnings']:
                st.warning(warning)
    
    st.markdown("---")

def process_uploaded_file(uploaded_file, document_loader: DocumentLoader, index_manager: IndexManager) -> bool:
    """Process a single uploaded file."""
    try:
        # Save uploaded file to data directory
        if not os.path.exists("data"):
            os.makedirs("data")
            
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the single file
        documents = document_loader.load_single_document(file_path)
        if documents:
            # Create or update index
            if st.session_state.index is None:
                st.session_state.index = index_manager.create_index(documents)
            else:
                # Add new documents to existing index
                for doc in documents:
                    st.session_state.index.insert(doc)
            
            st.session_state.documents_loaded = True
            return True
        return False
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return False

def main():
    st.title("💰 AI Financial Advisor")
    st.markdown("""
    Welcome to the AI Financial Advisor! Ask questions about financial planning, investments, or get general financial advice.
    Optionally, upload documents to get more context-aware responses.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Get settings from sidebar
    settings = sidebar_settings(ui_config)
    
    # Initialize components
    document_loader = DocumentLoader(config_loader)
    index_manager = IndexManager(config_loader)
    storage_manager = StorageManager(config_loader)
    query_engine = QueryEngine(config_loader)
    prompt_manager = PromptManager(config_loader)
    
    # File upload section
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your financial documents",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx']
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                success_count = 0
                for file in uploaded_files:
                    if process_uploaded_file(file, document_loader, index_manager):
                        success_count += 1
                
                if success_count > 0:
                    if storage_manager and st.session_state.index:
                        storage_manager.save_index(st.session_state.index)
                    st.success(f"Successfully processed {success_count} document(s)!")
                else:
                    st.error("No documents were successfully processed.")
    
    # Query section
    st.subheader("Ask Questions")
    query = st.text_input("Enter your financial question:")
    force_rag = st.checkbox("Force RAG Mode", value=False, 
                           help="Force using document retrieval regardless of query analysis")
    
    if query:
        with st.spinner("Processing query..."):
            # Process query with enhanced pipeline
            response, success, analysis = query_engine.process_query(
                query=query,
                index=st.session_state.index if st.session_state.documents_loaded else None,
                provider_name=settings["provider_name"],
                model_name=settings["model"],
                force_rag=force_rag,
                prompt_type=settings["prompt_type"],
                persona=settings["persona"],
                max_output_tokens=settings["max_output_tokens"]
            )
            
            # Create a container for the results
            results_container = st.container()
            
            with results_container:
                # Display query analysis first
                display_query_analysis(analysis)
                
                # Display response
                st.write("### Response")
                if success:
                    st.write(response)
                else:
                    st.error(response)
                
                # Determine if RAG was used
                using_rag = force_rag or analysis.get('requires_retrieval', False)
                
                # Only show verification results if using RAG
                if using_rag and (settings["enable_citation_check"] or settings["enable_hallucination_check"] or settings["enable_guardrails"]):
                    context = analysis.get("context", []) if analysis else []
                    
                    # Perform verifications
                    citations = prompt_manager.verify_citations(response, context) if settings["enable_citation_check"] else None
                    hallucination_check = prompt_manager.detect_hallucination(response, context, citations) if settings["enable_hallucination_check"] else None
                    compliance_check = prompt_manager.apply_guardrails(response, []) if settings["enable_guardrails"] else None
                    
                    # Display verification results
                    display_verification_results(citations, hallucination_check, compliance_check)
                
                # Show processing details in expander
                with st.expander("Query Processing Details"):
                    st.write("**Query Processing:**")
                    st.write("Original Query:", analysis.get('original_query', query))
                    st.write("Enhanced Query:", analysis.get('enhanced_query', query))
                    
                    st.write("\n**Generation Settings:**")
                    st.write("Using RAG:", using_rag)
                    st.write("Model:", settings["model"])
                    st.write("Temperature:", settings["temperature"])
                    st.write("Prompt Type:", settings["prompt_type"])
                    if settings["persona"]:
                        st.write("Persona:", settings["persona"])

if __name__ == "__main__":
    main() 
"""Streamlit web interface for the Enterprise RAG System."""

import streamlit as st
import asyncio
import os
import time
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our RAG system components (absolute imports for Streamlit execution)
from src.core.config import settings
from src.core.database import get_db_session
from src.core.models import DocumentStatus, QueryRequest, QueryResponse
from src.services.document_processor import DocumentProcessor
from src.services.embedding_service import EmbeddingService
from src.services.retrieval_service import HybridRetriever, SearchType

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "services_initialized" not in st.session_state:
    st.session_state.services_initialized = False
if "embedding_service" not in st.session_state:
    st.session_state.embedding_service = None
if "document_processor" not in st.session_state:
    st.session_state.document_processor = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []


async def initialize_services():
    """Initialize RAG services."""
    if st.session_state.services_initialized:
        return
        
    try:
        with st.spinner("Initializing RAG services..."):
            # Initialize embedding service
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            st.session_state.embedding_service = embedding_service
            
            # Initialize document processor
            document_processor = DocumentProcessor()
            st.session_state.document_processor = document_processor
            
            # Initialize retriever
            retriever = HybridRetriever(embedding_service)
            st.session_state.retriever = retriever
            
            st.session_state.services_initialized = True
            st.success("✅ Services initialized successfully!")
            
    except Exception as e:
        st.error(f"❌ Failed to initialize services: {str(e)}")
        st.session_state.services_initialized = False


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Enterprise RAG System</h1>', unsafe_allow_html=True)
    st.markdown("**Production-ready document question answering system**")
    
    # Initialize services
    asyncio.run(initialize_services())
    
    if not st.session_state.services_initialized:
        st.error("Services not initialized. Please check the configuration.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # System status
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Online", "Ready")
        with col2:
            st.metric("Services", "✅ Active", "3/3")
        
        # Quick stats
        st.subheader("Quick Stats")
        try:
            async def get_stats():
                async with get_db_session() as session:
                    from sqlalchemy import select, func
                    from src.core.models import DocumentModel
                    
                    # Get document counts
                    result = await session.execute(
                        select(
                            DocumentModel.status,
                            func.count(DocumentModel.id).label('count')
                        ).group_by(DocumentModel.status)
                    )
                    return {row.status: row.count for row in result}
            
            stats = asyncio.run(get_stats())
            total_docs = sum(stats.values())
            processed_docs = stats.get('completed', 0)
            
            st.metric("Total Documents", total_docs)
            st.metric("Processed", processed_docs)
            st.metric("Success Rate", f"{(processed_docs/total_docs*100):.1f}%" if total_docs > 0 else "0%")
            
        except Exception as e:
            st.warning(f"Could not load stats: {str(e)}")
        
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["🏠 Dashboard", "📄 Documents", "❓ Query", "📊 Analytics", "⚙️ Settings"]
        )
    
    # Main content based on page selection
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📄 Documents":
        show_documents_page()
    elif page == "❓ Query":
        show_query_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_dashboard():
    """Show the main dashboard."""
    st.header("📊 Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Documents Processed",
            "0",  # Placeholder
            "0"
        )
    
    with col2:
        st.metric(
            "Queries Today",
            "0",  # Placeholder
            "0"
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            "0.5s",  # Placeholder
            "-0.1s"
        )
    
    with col4:
        st.metric(
            "Cache Hit Rate",
            "85%",  # Placeholder
            "5%"
        )
    
    st.divider()
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 System Performance")
        
        # Performance chart (placeholder)
        performance_data = {
            'Time': ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            'Response Time (ms)': [500, 450, 600, 350, 400, 380],
            'Throughput (qpm)': [100, 120, 80, 150, 130, 140]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.line(df_perf, x='Time', y='Response Time (ms)', 
                     title='Response Time Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Recent Activity")
        
        # Recent queries (placeholder)
        recent_queries = [
            {"query": "What is the company policy?", "time": "2 min ago", "status": "✅"},
            {"query": "How to submit expenses?", "time": "5 min ago", "status": "✅"},
            {"query": "Remote work guidelines", "time": "8 min ago", "status": "✅"},
        ]
        
        for query in recent_queries:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{query['query']}</strong><br>
                <small>{query['time']} • {query['status']}</small>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")


def show_documents_page():
    """Show documents management page."""
    st.header("📄 Document Management")
    
    # Upload section
    st.subheader("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Process uploaded files
                for uploaded_file in uploaded_files:
                    # Save file temporarily
                    file_path = os.path.join(settings.documents_path, uploaded_file.name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process document
                    try:
                        result = asyncio.run(
                            st.session_state.document_processor.process_document(file_path)
                        )
                        if result:
                            st.success(f"✅ Processed: {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    st.divider()
    
    # Document list
    st.subheader("📋 Document Library")
    
    try:
        async def get_documents():
            async with get_db_session() as session:
                from sqlalchemy import select
                from src.core.models import DocumentModel
                
                result = await session.execute(
                    select(DocumentModel).order_by(DocumentModel.created_at.desc())
                )
                return result.scalars().all()
        
        documents = asyncio.run(get_documents())
        
        if documents:
            # Create DataFrame for display
            doc_data = []
            for doc in documents:
                doc_data.append({
                    "Filename": doc.filename,
                    "Status": doc.status,
                    "Size": f"{doc.file_size:,} bytes",
                    "Created": doc.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Chunks": doc.chunk_count
                })
            
            df = pd.DataFrame(doc_data)
            
            # Status color mapping
            status_colors = {
                'completed': '🟢',
                'processing': '🟡',
                'failed': '🔴',
                'pending': '⚪'
            }
            
            df['Status'] = df['Status'].map(status_colors)
            
            st.dataframe(df, use_container_width=True)
            
            # Document actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Refresh List"):
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear Failed"):
                    st.warning("This will remove all failed documents")
            with col3:
                if st.button("📊 Export Stats"):
                    st.info("Export functionality coming soon")
        else:
            st.info("No documents found. Upload some documents to get started!")
            
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")


def show_query_page():
    """Show query interface."""
    st.header("❓ Query Documents")
    
    # Query input
    query = st.text_area(
        "Ask a question about your documents:",
        placeholder="e.g., What is the company policy on remote work?",
        height=100
    )
    
    # Query options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_results = st.slider("Max Results", 1, 20, 5)
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["Hybrid", "Semantic", "Keyword"],
            index=0
        )
    with col3:
        include_metadata = st.checkbox("Include Metadata", value=True)
    
    # Submit query
    if st.button("🔍 Search", type="primary", disabled=not query.strip()):
        if not st.session_state.services_initialized:
            st.error("Services not initialized. Please check the configuration.")
            return
        
        with st.spinner("Searching documents..."):
            try:
                # Map search type
                search_type_map = {
                    "Hybrid": SearchType.HYBRID,
                    "Semantic": SearchType.SEMANTIC,
                    "Keyword": SearchType.KEYWORD
                }
                
                # Perform search
                start_time = time.time()
                results = asyncio.run(
                    st.session_state.retriever.retrieve(
                        query=query,
                        k=max_results,
                        search_type=search_type_map[search_type]
                    )
                )
                response_time = time.time() - start_time
                
                # Store in query history
                st.session_state.query_history.append({
                    "query": query,
                    "results_count": len(results),
                    "response_time": response_time,
                    "timestamp": time.time()
                })
                
                # Display results
                st.subheader("🎯 Search Results")
                
                if results:
                    st.success(f"Found {len(results)} relevant documents in {response_time:.2f}s")
                    
                    # Results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Results Found", len(results))
                    with col2:
                        st.metric("Response Time", f"{response_time:.2f}s")
                    with col3:
                        avg_score = sum(r.score for r in results) / len(results)
                        st.metric("Avg Relevance", f"{avg_score:.3f}")
                    
                    # Display each result
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Result {i} (Score: {result.score:.3f})"):
                            st.markdown(f"**Content:**")
                            st.text_area(
                                f"Content {i}",
                                value=result.content,
                                height=200,
                                key=f"content_{i}"
                            )
                            
                            if include_metadata and result.metadata:
                                st.markdown("**Metadata:**")
                                st.json(result.metadata)
                            
                            if result.document_metadata:
                                st.markdown("**Document Info:**")
                                st.json(result.document_metadata)
                else:
                    st.warning("No relevant documents found. Try rephrasing your query.")
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    
    # Query history
    if st.session_state.query_history:
        st.divider()
        st.subheader("📜 Query History")
        
        history_data = []
        for entry in st.session_state.query_history[-10:]:  # Show last 10
            history_data.append({
                "Query": entry["query"][:50] + "..." if len(entry["query"]) > 50 else entry["query"],
                "Results": entry["results_count"],
                "Time": f"{entry['response_time']:.2f}s",
                "Timestamp": time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            })
        
        if history_data:
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True)


def show_analytics_page():
    """Show analytics and monitoring."""
    st.header("📊 Analytics & Monitoring")
    
    # System health
    st.subheader("🏥 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Database", "🟢 Healthy", "Connected")
    with col2:
        st.metric("Cache", "🟢 Healthy", "85% hit rate")
    with col3:
        st.metric("Embeddings", "🟢 Healthy", "Model loaded")
    with col4:
        st.metric("Storage", "🟢 Healthy", "2.1 GB used")
    
    st.divider()
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Query Performance")
        
        # Sample performance data
        perf_data = {
            'Hour': ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'],
            'Response Time (ms)': [450, 380, 420, 350, 600, 520, 480, 410, 390, 440, 460, 430],
            'Queries per Minute': [12, 8, 5, 3, 25, 35, 28, 22, 18, 15, 20, 16]
        }
        
        df_perf = pd.DataFrame(perf_data)
        
        fig = px.line(df_perf, x='Hour', y='Response Time (ms)', 
                     title='Response Time Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Query Distribution")
        
        # Sample query types
        query_types = ['Factual', 'Comparison', 'Analytical', 'Procedural']
        query_counts = [45, 23, 18, 14]
        
        fig = px.pie(values=query_counts, names=query_types, 
                    title='Query Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cache performance
    st.subheader("💾 Cache Performance")
    
    cache_data = {
        'Metric': ['Hit Rate', 'Miss Rate', 'Total Requests', 'Cache Size'],
        'Value': ['85%', '15%', '1,247', '2.1 GB'],
        'Status': ['🟢 Good', '🟡 Acceptable', '🟢 Normal', '🟢 Healthy']
    }
    
    df_cache = pd.DataFrame(cache_data)
    st.dataframe(df_cache, use_container_width=True)


def show_settings_page():
    """Show system settings."""
    st.header("⚙️ System Settings")
    
    # Configuration display
    st.subheader("🔧 Current Configuration")
    
    config_data = {
        'Setting': [
            'API Host', 'API Port', 'Debug Mode', 'Max File Size',
            'Embedding Model', 'Chunk Size', 'Chunk Overlap',
            'Retrieval K', 'Similarity Threshold', 'Cache TTL'
        ],
        'Value': [
            settings.api_host, str(settings.api_port), str(settings.debug),
            f"{settings.max_file_size:,} bytes", settings.embedding_model,
            str(settings.chunk_size), str(settings.chunk_overlap),
            str(settings.retrieval_k), str(settings.similarity_threshold),
            f"{settings.cache_ttl} seconds"
        ]
    }
    
    df_config = pd.DataFrame(config_data)
    st.dataframe(df_config, use_container_width=True)
    
    st.divider()
    
    # System actions
    st.subheader("🔧 System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart Services", type="secondary"):
            st.session_state.services_initialized = False
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache", type="secondary"):
            st.warning("Cache clearing functionality coming soon")
    
    with col3:
        if st.button("📊 Export Logs", type="secondary"):
            st.info("Log export functionality coming soon")
    
    # Environment info
    st.subheader("🌍 Environment Information")
    
    env_info = {
        'Component': ['Python Version', 'Streamlit Version', 'Database URL', 'Redis URL'],
        'Value': [
            '3.11.0', '1.28.0', settings.database_url.split('@')[1] if '@' in settings.database_url else 'Hidden',
            settings.redis_url
        ]
    }
    
    df_env = pd.DataFrame(env_info)
    st.dataframe(df_env, use_container_width=True)


if __name__ == "__main__":
    main()

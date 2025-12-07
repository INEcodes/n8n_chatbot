"""
Gemini-Powered RAG Chatbot with Streamlit
This script creates a web-based chatbot interface using Streamlit
that retrieves documents from Pinecone and generates responses using Google's Gemini API.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 2. SETUP AND CONFIGURATION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Set your API keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    st.error("❌ Missing required API keys in environment variables")
    st.stop()

# Configure Google Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Pinecone configuration
PINECONE_CLOUD = os.environ.get('PINECONE_CLOUD') or 'aws'
PINECONE_REGION = os.environ.get('PINECONE_REGION') or 'us-east-1'
INDEX_NAME = "project"
NAMESPACE = "_default_"
MODEL_NAME = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

# ============================================================================
# 3. GEMINI EMBEDDING FUNCTION
# ============================================================================

def create_gemini_embedding(text, model_name=MODEL_NAME):
    """
    Create embedding using Google's Gemini embedding model.
    
    Args:
        text: Text string to embed
        model_name: Gemini embedding model to use
        
    Returns:
        Embedding vector
    """
    try:
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        return None

# ============================================================================
# 4. PINECONE RETRIEVAL
# ============================================================================

def retrieve_relevant_documents(query, index, top_k=3, namespace=NAMESPACE):
    """
    Retrieve relevant documents from Pinecone based on query.
    
    Args:
        query: User query string
        index: Pinecone index object
        top_k: Number of top results to retrieve
        namespace: Namespace to search in
        
    Returns:
        List of retrieved documents with metadata
    """
    # Create embedding for query
    query_embedding = create_gemini_embedding(query)
    
    if query_embedding is None:
        return []
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    
    documents = []
    for match in results['matches']:
        # Extract text from various possible metadata fields
        text_content = (
            match['metadata'].get('text_snippet', '') or 
            match['metadata'].get('text', '') or 
            match['metadata'].get('description', '') or
            match['metadata'].get('title', '')
        )
        
        documents.append({
            'id': match['id'],
            'score': match['score'],
            'text': text_content,
            'title': match['metadata'].get('title', ''),
            'url': match['metadata'].get('url', ''),
            'description': match['metadata'].get('description', ''),
            'metadata': match['metadata']
        })
    
    return documents

# ============================================================================
# 5. GEMINI RESPONSE GENERATION
# ============================================================================

def generate_response(query, retrieved_docs, model=CHAT_MODEL):
    """
    Generate response using Gemini based on retrieved documents.
    
    Args:
        query: User query
        retrieved_docs: List of retrieved documents
        model: Gemini model to use for generation
        
    Returns:
        Generated response string
    """
    # Build context from retrieved documents
    context = "Based on the following information:\n\n"
    
    for i, doc in enumerate(retrieved_docs, 1):
        context += f"[Document {i}]\n"
        
        # Include title if available
        if doc.get('title'):
            context += f"Title: {doc['title']}\n"
        
        # Include the main text content
        context += f"Content: {doc['text']}\n"
        
        # Include URL if available
        if doc.get('url'):
            context += f"Source: {doc['url']}\n"
        
        context += "\n"
    
    # Create system prompt
    system_prompt = """You are a helpful assistant that answers questions based only on the provided context. 
If the context doesn't contain information to answer the question, politely say so.
Keep answers concise and relevant to the question.
If URLs are provided in the context, you can mention them as sources."""
    
    # Create the full prompt
    full_prompt = f"""{system_prompt}

Context:
{context}

User Question: {query}

Answer:"""
    
    try:
        # Generate response using Gemini with updated API
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500
            )
        )
        
        return response.text if response.text else "Unable to generate response"
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ============================================================================
# 6. STREAMLIT SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def init_chatbot():
    """Initialize chatbot and cache it."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        return index
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {e}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = init_chatbot()

# ============================================================================
# 7. STREAMLIT UI
# ============================================================================

# Title and description
st.title("🤖 RAG Chatbot with Gemini")
st.markdown("Ask questions and get answers based on your knowledge base")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    top_k = st.slider(
        "Number of documents to retrieve:",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher values retrieve more context but may be slower"
    )
    
    temperature = st.slider(
        "Response creativity (Temperature):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make responses more creative, lower values more factual"
    )
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### ℹ️ How to use:")
    st.markdown("""
    1. Type your question in the input box below
    2. The chatbot will search relevant documents
    3. Get answers based on your knowledge base
    4. View sources and confidence scores
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "docs" in message:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["docs"], 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**[{i}] {doc.get('title', 'No title')}**")
                        st.caption(f"Score: {doc['score']:.3f}")
                    with col2:
                        if doc.get('url'):
                            st.markdown(f"[🔗 Link]({doc['url']})")

# Chat input
if st.session_state.pinecone_index is not None:
    user_input = st.chat_input("Ask me anything...")
    
    if user_input:
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving documents and generating response..."):
                # Retrieve documents
                retrieved_docs = retrieve_relevant_documents(
                    user_input,
                    st.session_state.pinecone_index,
                    top_k=top_k,
                    namespace=NAMESPACE
                )
                
                # Generate response
                if not retrieved_docs:
                    response = "I couldn't find relevant information in the knowledge base to answer your question."
                else:
                    response = generate_response(user_input, retrieved_docs)
                
                st.markdown(response)
                
                # Display sources
                if retrieved_docs:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**[{i}] {doc.get('title', 'No title')}**")
                                st.caption(f"Relevance Score: {doc['score']:.3f}")
                                if doc.get('url'):
                                    st.markdown(f"📄 [Source Link]({doc['url']})")
                            with col2:
                                st.metric("Match", f"{doc['score']*100:.1f}%")
        
        # Add to session state with docs
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "docs": retrieved_docs
        })

else:
    st.error("❌ Failed to connect to Pinecone. Please check your API keys.")
import streamlit as st
import os
from pdf2image import convert_from_path
from PIL import Image
import io
import base64
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Create constants and required directories
PDF_STORAGE_PATH = 'document_store/pdfs/'
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# Enhanced UI Styling
st.markdown("""
    <style>
    /* Main App Styling */
    .stApp {
        background: linear-gradient(to bottom right, #0E1117, #1A1C24);
        color: #FFFFFF;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #00FFAA 0%, #00B8FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    
    .sub-header {
        color: #B0B0B0 !important;
        font-size: 1.2rem !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    /* Chat Input Styling */
    .stChatInput {
        border-radius: 12px !important;
    }
    
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 2px solid #3A3A3A !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
    }
    
    .stChatInput input:focus {
        border-color: #00FFAA !important;
        box-shadow: 0 0 15px rgba(0, 255, 170, 0.2) !important;
    }
    
    /* Message Styling */
    .stChatMessage[data-testid="stChatMessage"] {
        padding: 1rem !important;
        margin: 1rem 0 !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.2s ease;
    }
    
    /* User Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background: linear-gradient(145deg, #1E1E1E, #2A2A2A) !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* Assistant Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background: linear-gradient(145deg, #2A2A2A, #363636) !important;
        border: 1px solid #404040 !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
    }
    
    /* File Uploader Styling */
    .stFileUploader {
        background: linear-gradient(145deg, #1E1E1E, #2A2A2A) !important;
        border: 2px dashed #3A3A3A !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        margin: 1.5rem 0 !important;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #00FFAA !important;
        box-shadow: 0 0 20px rgba(0, 255, 170, 0.1) !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(90deg, #00FFAA, #00B8FF) !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 255, 170, 0.3) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: #00FFAA !important;
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
        border-left: 5px solid #00FFAA;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-10px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Spinner */
    .stSpinner {
        border-color: #00FFAA !important;
    }
    
    /* Thinking Process Container */
    .thinking-process {
        background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
        border-left: 5px solid #00B8FF;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #B0B0B0;
    }
    
    /* Answer Container */
    .answer-container {
        background: linear-gradient(145deg, #2A2A2A, #363636);
        border: 1px solid #404040;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        width: 100%;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Make content wider */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Enhance code block readability */
    code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        display: block;
        padding: 1rem !important;
        background: #1E1E1E !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

def save_uploaded_file(uploaded_file):
    try:
        # Ensure directory exists
        os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
        
        # Use os.path.join for proper path handling
        file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        raise

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

def get_page_text(file_path, page_number):
    """Extract text from a specific page of the PDF."""
    try:
        pdf = PDFPlumberLoader(file_path).load()
        if 0 <= page_number < len(pdf):
            return pdf[page_number].page_content
        return None
    except Exception as e:
        return None

# Add these new functions for PDF image handling
def convert_pdf_page_to_image(pdf_path, page_number):
    """Convert a specific PDF page to an image."""
    try:
        # For Windows: Specify poppler path if not in PATH
        import platform
        if platform.system() == "Windows":
            try:
                # Try with default PATH first
                images = convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1)
            except Exception:
                # If failed, try with common Poppler installation paths
                common_paths = [
                    r"C:\Program Files\poppler-23.11.0\Library\bin",
                    r"C:\Program Files\poppler-23.11.0\bin",
                    r"C:\poppler-23.11.0\bin",
                    r"C:\poppler\bin"
                ]
                
                for poppler_path in common_paths:
                    if os.path.exists(poppler_path):
                        images = convert_from_path(
                            pdf_path,
                            first_page=page_number + 1,
                            last_page=page_number + 1,
                            poppler_path=poppler_path
                        )
                        return images[0] if images else None
                
                # If still not found, show a helpful error message
                st.error("""
                    Poppler is not installed or not found. Please follow these steps:
                    1. Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
                    2. Extract the zip file
                    3. Add the 'bin' folder path to your system's PATH environment variable
                    
                    Alternatively, you can continue using the app without page previews.
                """)
                return None
        else:
            # For non-Windows systems
            images = convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1)
        
        return images[0] if images else None
    except Exception as e:
        st.warning("Page preview not available. Continuing with text-only display.")
        return None

def get_image_base64(image):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_page_preview(pdf_path, page_number, width=800):
    """Create a preview image of a PDF page with specified width."""
    try:
        image = convert_pdf_page_to_image(pdf_path, page_number)
        if image:
            # Calculate height to maintain aspect ratio
            aspect_ratio = image.height / image.width
            height = int(width * aspect_ratio)
            
            # Resize image
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            return get_image_base64(image)
        return None
    except Exception as e:
        st.warning("Unable to create page preview. Continuing with text-only display.")
        return None

# Updated UI Section with new title and subtitle
st.markdown('<h1 class="main-header">🧠 PDFIntellect DeepSeek R1</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Unleash the Power of AI-Driven Document Intelligence</p>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_pdf = st.file_uploader(
        "📄 Upload Your Research Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )

if uploaded_pdf:
    try:
        saved_path = save_uploaded_file(uploaded_pdf)
        
        if os.path.exists(saved_path):
            raw_docs = load_pdf_documents(saved_path)
            processed_chunks = chunk_documents(raw_docs)
            index_documents(processed_chunks)
            
            st.markdown("""
                <div class="success-message">
                    ✨ Document processed successfully! Your AI assistant is ready to answer your questions.
                </div>
            """, unsafe_allow_html=True)
            
            user_input = st.chat_input("💭 Ask anything about your document...")
            
            if user_input:
                with st.chat_message("user", avatar="👤"):
                    st.write(user_input)
                
                with st.spinner("🤔 Analyzing document..."):
                    # Show thinking process
                    st.markdown("""
                        <div class="thinking-process">
                            🔍 Searching through document...<br>
                            📝 Analyzing relevant sections...<br>
                            🧩 Synthesizing information...
                        </div>
                    """, unsafe_allow_html=True)
                    
                    relevant_docs = find_related_documents(user_input)
                    ai_response = generate_answer(user_input, relevant_docs)
                    
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown("""
                        <div class="answer-container">
                            {}
                        </div>
                    """.format(ai_response), unsafe_allow_html=True)
                    
                # Enhanced source context display with better formatting
                with st.expander("📚 View Source Context"):
                    st.markdown("""
                        <div style='background: #1E1E1E; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                            <h3 style='color: #00FFAA; margin-bottom: 10px;'>🎯 Context Sources Used</h3>
                            <p style='color: #B0B0B0;'>The AI used the following sections from your document to generate the response:</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for i, doc in enumerate(relevant_docs, 1):
                        # Create a clean container for each source
                        st.markdown("---")
                        col1, col2 = st.columns([1, 11])
                        
                        with col1:
                            st.markdown(f"<h3 style='color: #00FFAA;'>📄</h3>", unsafe_allow_html=True)
                        
                        with col2:
                            # Source header
                            st.markdown(f"""
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h3 style='color: #00FFAA; margin: 0;'>Source {i}</h3>
                                    <span style='background: #00FFAA; color: #1E1E1E; padding: 5px 10px; border-radius: 15px; font-size: 0.8em;'>
                                        Page {doc.metadata.get('page', 0) + 1}
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Source content
                            st.markdown("""
                                <div style='
                                    background: #2A2A2A;
                                    border-radius: 10px;
                                    padding: 15px;
                                    margin: 10px 0;
                                    font-family: monospace;
                                    white-space: pre-wrap;
                                    color: #E0E0E0;
                                '>
                                {}
                                </div>
                            """.format(doc.page_content), unsafe_allow_html=True)
                            
                            # Metadata
                            st.markdown(f"""
                                <div style='
                                    background: #1E1E1E;
                                    border-radius: 10px;
                                    padding: 10px;
                                    font-size: 0.9em;
                                    color: #B0B0B0;
                                '>
                                    <strong>📍 Location:</strong> Character range {doc.metadata.get('start_index', 'N/A')} - {doc.metadata.get('end_index', 'N/A')}
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.error("File was not saved properly. Please try again.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the document: {str(e)}")

# Updated footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        Made with 🧠 by NexusMinds Team
    </div>
""", unsafe_allow_html=True)
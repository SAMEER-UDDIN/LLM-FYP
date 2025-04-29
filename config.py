import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory - use the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
CHROMA_INDEX_PATH = os.path.join(BASE_DIR, "chroma_embeddings")
FOLDER_PATH = os.path.join(BASE_DIR, "goofiya data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# RAG Configuration
RETRIEVER_SEARCH_DISTANCE = 0.5  # Similarity threshold for retrieval
RETRIEVER_K = 5  # Number of documents to retrieve

# Text Processing Configuration
CHUNK_SIZE = 20  # Number of sentences per chunk
CHUNK_OVERLAP = 5  # Overlap between chunks

# LLM Configuration
# Groq API (Llama-3.1-70b-Instructy)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Get from environment variable
LLM_MODEL_NAME = "llama3-8b-8192"  # Fallback model if premium not selected
PREMIUM_LLM_MODEL_NAME = "llama-3.3-70b-versatile"  # Use premium model for better quality

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# LLM Generation Parameters
TEMPERATURE = 0.7    # Higher for more creative, lower for more deterministic
TOP_P = 0.9         # Controls diversity

# System Prompts
SYSTEM_PROMPT_CHAT = """You are a knowledgeable assistant specializing in pharmaceutical procedures, 
regulations, and documentation. Your task is to provide accurate, clear, and concise responses based on 
the context provided. Maintain a professional tone while being helpful and informative.

Your capabilities:
- Explaining pharmaceutical procedures and protocols
- Clarifying regulatory requirements
- Helping with document interpretation
- Providing step-by-step guidance on processes
Greet the user and ignore the context provided belwo and ask how you can assist them today, If the user greets.
Always base your answers on the provided context. If you're unsure or the context doesn't contain 
relevant information, admit that and suggest what might help instead of making up information.
Also don't answer irrelevant questions. (e.g. "What is cat?")"""

SYSTEM_PROMPT_REPORT = """You are a pharmaceutical documentation specialist tasked with creating comprehensive,
detailed reports based on the provided context. Your reports should be thorough, well-structured, and professionally written. 
Create your reports with the given structure. Your report should be approximately 3 pages in length, 
comprehensive yet focused on the specific query. Use precise language, industry-standard terminology, and maintain a formal tone throughout."""

# Supported file types for upload
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx"]

# UI Configuration
APP_TITLE = "Pharma RAG"
APP_LAYOUT = "wide"
APP_THEME = "dark"

# Ensure necessary directories exist
for directory in [CHROMA_INDEX_PATH, UPLOAD_FOLDER]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

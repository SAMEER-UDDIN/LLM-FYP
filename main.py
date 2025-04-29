from config import FOLDER_PATH, CHROMA_INDEX_PATH, RETRIEVER_SEARCH_DISTANCE, RETRIEVER_K, GROQ_API_KEY
from data_loader import read_txts_from_folder
from vector_store import initialize_vector_store
from chatbot import get_bot_response
import ui
import os
from langchain_chroma import Chroma
from vector_store import get_embedding_function



def initialize_system():
    """
    Initialize the RAG system components:
    1. Check API key availability
    2. Load data from files
    3. Initialize vector store with the loaded data
    4. Create a retriever from the vector store
    
    Returns:
        retriever: Configured document retriever
    """
    print("Starting system initialization...")
    
    # Check for API key in environment
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not found in environment. Please set this variable.")
        print("Create a .env file with GROQ_API_KEY=your_api_key")
    
    try:
        # Check if embeddings already exist
        chroma_exists = os.path.exists(CHROMA_INDEX_PATH) and os.listdir(CHROMA_INDEX_PATH)
        
        if chroma_exists:
            print(f"Embeddings found at {CHROMA_INDEX_PATH}")
            # Use existing vector store without adding new documents
            chroma_vector_store = initialize_vector_store(use_existing=True)
        else:
            # Load data from files and create new vector store
            print(f"Loading data from {FOLDER_PATH}")
            file_data = read_txts_from_folder(FOLDER_PATH)
            print(f"Loaded {len(file_data)} files")
            
            # Initialize vector store with the loaded data
            chroma_vector_store = initialize_vector_store(file_data)
        
        if chroma_vector_store is None:
            raise ValueError("Vector store initialization failed, returned None")
        
        # Create retriever with configured parameters
        retriever = chroma_vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_K},
            search_type="similarity",
            search_distance=RETRIEVER_SEARCH_DISTANCE
        )
        print("Retriever initialized successfully")
        
        return retriever
    
    except Exception as e:
        import traceback
        print(f"Error during system initialization: {str(e)}")
        print("Detailed traceback:")
        traceback.print_exc()
        
        # In case of failure, try creating a minimal functional retriever
        try:
            print("Attempting to create fallback vector store...")
            # Make sure the directory exists
            os.makedirs(CHROMA_INDEX_PATH, exist_ok=True)
            
            # Try to create a basic vector store with default embeddings
            fallback_store = Chroma(
                persist_directory=CHROMA_INDEX_PATH,
                embedding_function=get_embedding_function()
            )
            
            # Create a basic retriever
            fallback_retriever = fallback_store.as_retriever(
                search_kwargs={"k": 1},
                search_type="similarity"
            )
            
            print("Created fallback retriever")
            return fallback_retriever
        except Exception as fallback_error:
            print(f"Failed to create fallback retriever: {str(fallback_error)}")
            raise e  # Re-raise the original error

def main():
    """
    Main entry point for the application:
    1. Pass the initialization function to the UI
    2. Start the Streamlit UI
    """
    # Pass the function itself, don't call it here
    # retriever = initialize_system() 
    
    # Start the UI, passing the initialization function
    ui.run(initialize_system, get_bot_response) # Pass the function

if __name__ == "__main__":
    main()
  
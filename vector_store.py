import os
import uuid
from langchain_chroma import Chroma
from config import CHROMA_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from text_utils import clean_text, SimpleSentenceSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(quiet=False):
    """Initializes and returns a LangChain-compatible embedding function."""
    if not quiet:
        print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)



def initialize_vector_store(file_data=None, use_existing=False, quiet=False):
    """
    Initialize or load the vector store and add documents from file_data.
    
    This function follows these steps:
    1. Initialize embedding function
    2. If use_existing=True or no file_data is provided, load existing vector store
    3. Otherwise:
       - Process each file (clean text, split into chunks)
       - Create metadata and IDs
       - Add documents to vector store (new or existing)
    
    Args:
        file_data (list, optional): List of dictionaries with file_name and content
        use_existing (bool): Whether to use an existing vector store without adding new documents
        quiet (bool): If True, suppresses informational messages
        
    Returns:
        Chroma: Initialized Chroma vector store
    """
    try:
        # Get the embedding function
        embedding_function = get_embedding_function(quiet=quiet)
        
        # Check if embeddings directory exists
        chroma_exists = os.path.exists(CHROMA_INDEX_PATH) and os.listdir(CHROMA_INDEX_PATH)
        
        # Case 1: Use existing vector store if specified or if no file_data and store exists
        if use_existing and chroma_exists:
            if not quiet:
                print(f"Loading existing Chroma vector store from {CHROMA_INDEX_PATH}")
            return Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_function)
            
        # Case 2: No file data provided and no existing store or not using existing
        if file_data is None:
            if chroma_exists:
                if not quiet:
                    print(f"Loading existing Chroma vector store from {CHROMA_INDEX_PATH}")
                return Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_function)
            else:
                if not quiet:
                    print("Creating new empty Chroma vector store")
                os.makedirs(CHROMA_INDEX_PATH, exist_ok=True)
                return Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_function)
        
        # Case 3: Process documents if file_data is provided
        if not quiet:
            print(f"Processing {len(file_data) if file_data else 0} files for vector store")
        
        # Initialize text splitter
        splitter = SimpleSentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        documents, metadatas, ids = [], [], []
        doc_counter = 0
        
        # Process each file
        for data in file_data:
            file_name = data.get('file_name', 'unknown')
            content = data.get('content', '')
            
            if not content:
                if not quiet:
                    print(f"Warning: Empty content for file {file_name}")
                continue
                
            # Clean text
            cleaned_content = clean_text(content)
            
            # Split into chunks
            text_chunks = splitter.split_text(cleaned_content)
            
            if text_chunks:
                if not quiet:
                    print(f"Processing {file_name}: Created {len(text_chunks)} chunks")
                
                # Create metadata and IDs for each chunk
                for i, chunk in enumerate(text_chunks):
                    chunk_id = f"doc_{doc_counter + i}"
                    documents.append(chunk)
                    metadatas.append({
                        'source': file_name,
                        'chunk': i + 1,
                        'total_chunks': len(text_chunks)
                    })
                    ids.append(chunk_id)
                    
                doc_counter += len(text_chunks)
            else:
                if not quiet:
                    print(f"Warning: No chunks created for {file_name}")
    
        # Check if we have documents to add
        if not documents:
            if not quiet:
                print("Warning: No documents found to add to the vector store")
            os.makedirs(CHROMA_INDEX_PATH, exist_ok=True)
            return Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_function)
    
        # Determine whether to create new vector store or update existing one
        if chroma_exists:
            if not quiet:
                print(f"Loading existing Chroma vector store from {CHROMA_INDEX_PATH}")
            chroma_vector_store = Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_function) 
            
            if not quiet:
                print(f"Adding {len(documents)} new document chunks")
            chroma_vector_store.add_texts(
                texts=documents,
                metadatas=metadatas,
                ids=ids
            )
        else:
            if not quiet:
                print(f"Creating new Chroma vector store at {CHROMA_INDEX_PATH}")
            os.makedirs(CHROMA_INDEX_PATH, exist_ok=True)
            chroma_vector_store = Chroma.from_texts(
                texts=documents,
                embedding=embedding_function,
                metadatas=metadatas,
                ids=ids,
                persist_directory=CHROMA_INDEX_PATH
            )
        
        if not quiet:
            print(f"Vector store persisted with {len(documents)} document chunks")
        return chroma_vector_store
        
    except Exception as e:
        print(f"Error in initialize_vector_store: {str(e)}")
        # Create a basic empty store for recovery
        os.makedirs(CHROMA_INDEX_PATH, exist_ok=True)
        return Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_function)

def add_document_to_store(document_data, retriever=None):
    """
    Add a single document to an existing vector store
    
    Args:
        document_data (dict): Dictionary with file_name and content
        retriever: Optional retriever object to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the embedding function
        embedding_function = get_embedding_function()

        # Initialize text splitter
        splitter = SimpleSentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        file_name = document_data.get('file_name', 'uploaded_file')
        content = document_data.get('content', '')
        
        if not content:
            print(f"Warning: Empty content for file {file_name}")
            return False
            
        # Clean text
        cleaned_content = clean_text(content)
        
        # Split into chunks
        text_chunks = splitter.split_text(cleaned_content)
        
        if not text_chunks:
            print(f"Warning: No chunks created for {file_name}")
            return False
            
        print(f"Processing {file_name}: Created {len(text_chunks)} chunks")
        
        # Prepare documents, metadata, and IDs
        documents, metadatas, ids = [], [], []
        
        # Get the next available doc ID
        prefix = str(uuid.uuid4())[:8]
        
        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{prefix}_{i}"
            documents.append(chunk)
            metadatas.append({
                'source': file_name,
                'chunk': i + 1,
                'total_chunks': len(text_chunks),
                'added': 'manual_upload'
            })
            ids.append(chunk_id)
        
        # Load the existing vector store
        if os.path.exists(CHROMA_INDEX_PATH):
             # Pass embedding_function when loading
            chroma_vector_store = Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_function)
            
            # Add the new document chunks
            print(f"Adding {len(documents)} chunks from {file_name} to vector store")
            chroma_vector_store.add_texts(
                texts=documents,
                metadatas=metadatas,
                ids=ids
            )
            

            print(f"Vector store updated with new document: {file_name}")
            return True
        else:
            print("Error: Vector store does not exist")
            return False
            
    except Exception as e:
        print(f"Error adding document to store: {str(e)}")
        return False

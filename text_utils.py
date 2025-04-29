import re

def clean_text(text):
    """
    Clean text by removing extra whitespace and form feeds.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with a single space
    text = re.sub(r'[\x0c]', ' ', text)  # Remove form feed characters
    return text.strip()


class SimpleSentenceSplitter:
    """
    Split text into chunks based on sentences.
    
    This is optimized for RAG applications where we want manageable chunks
    that preserve context but are small enough for effective embedding.
    """
    
    def __init__(self, chunk_size=20, chunk_overlap=5):
        """
        Initialize the sentence splitter.
        
        Args:
            chunk_size (int): Number of sentences per chunk
            chunk_overlap (int): Number of sentences to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        """
        Split text into chunks of sentences.
        
        Args:
            text (str): The text to split
            
        Returns:
            list: List of text chunks
        """
        # Improved regex for better sentence splitting - matches period, exclamation, question mark 
        # followed by space and capital letter
        sentences = re.split(r'(?<=[.!?]) +(?=[A-Z])', text)
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        chunks = []
        
        # More robust chunking logic with proper overlap handling
        start_index = 0
        while start_index < len(sentences):
            end_index = min(start_index + self.chunk_size, len(sentences))
            chunk_sentences = sentences[start_index:end_index]
            chunk = " ".join(chunk_sentences).strip()
            
            # Only add non-empty chunks
            if chunk:
                chunks.append(chunk)
            
            # Move start index for the next chunk, considering overlap
            step = max(1, self.chunk_size - self.chunk_overlap)  # Ensure step is at least 1
            start_index += step
        
        return chunks
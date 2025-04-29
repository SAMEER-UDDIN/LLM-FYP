import re
import groq
import time # Add time import for potential delays if needed
from config import (
    GROQ_API_KEY, 
    PREMIUM_LLM_MODEL_NAME,
    LLM_MODEL_NAME, 
    TEMPERATURE,
    TOP_P,
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_REPORT
)
# Update imports for the newer LangChain version
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

# Initialize Groq client
client = groq.Groq(api_key=GROQ_API_KEY)

# Create a simple chat message history implementation
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []
        
    def add_message(self, message):
        self.messages.append(message)
        
    def clear(self):
        self.messages = []

# Initialize a global chat history store
chat_histories = {}

# Maximum number of messages to include in chat history for context
MAX_HISTORY_MESSAGES = 4

# Maximum token estimate for context
MAX_CONTEXT_TOKENS = 3000

def get_chat_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

def generate_chat_prompt(context, user_query):
    """
    Generate a standard chat prompt for the LLM.
    
    Args:
        context (str): The retrieved context from the vector store
        user_query (str): The user's original query
        
    Returns:
        dict: Messages formatted for Groq API
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CHAT},
        {"role": "user", "content": f"""Answer the general user queries and greet properly, ignore the context below if user asks a general question or greets.
         Else, Answer the following question based on the provided context. Don't add any extra or false information.
        
Context:
{context}

Question: {user_query}"""}
    ]
    return messages

def generate_report_prompt(context, user_query):
    """
    Generate a detailed report prompt for the LLM.
    
    Args:
        context (str): The retrieved context from the vector store
        user_query (str): The user's original query
        
    Returns:
        dict: Messages formatted for Groq API
    """
    # Initial report prompt (concise version, removed continuation marker instruction)
    enhanced_report_prompt = f"""Generate an extremely detailed, comprehensive, professional report based on the query and context below. Use beautiful Markdown formatting.

Context:
{context}

Query: {user_query}

Report Structure:

# [Informative Title]

## Executive Summary (min 300 words): Comprehensive overview, key findings, critical details, recommendations.

## Introduction (min 250 words): Context, purpose, background, relevance.

## Scope (min 200 words): Coverage, limitations, specific procedures/regulations addressed.

## Methodology (min 200 words): Information sourcing, analysis, synthesis, references to specific documents.

## Findings (min 800-1000 words): MOST SUBSTANTIAL SECTION. Present ALL relevant info with extensive detail. Use subsections, lists, bullet points. Include procedures, steps, regulations, technical details, parameters, specifications. Explain concepts and connections. Highlight critical points.

## Analysis (min 300-400 words): Assess findings, evaluate information completeness, identify strengths/gaps, compare to best practices, discuss challenges, assess compliance.

## Recommendations (min 300 words): Detailed, actionable steps based on findings. Suggest improvements, controls, documentation changes. Prioritize.

## Conclusion (min 200 words): Summarize key findings, implications, path forward.

## References: List all sources (SOPs, guidelines, regulations).

---

FORMATTING & INSTRUCTIONS:
- Use proper Markdown (headings, subheadings, bold, lists, tables if needed, blockquotes for references).
- Be highly detailed and comprehensive (target 8-10 printed pages).
""" # Removed the [REPORT_CONTINUES] instruction
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REPORT},
        {"role": "user", "content": enhanced_report_prompt}
    ]
    return messages

def truncate_context(context, max_tokens=MAX_CONTEXT_TOKENS):
    """
    Truncate context to an approximate token limit.
    Using rough estimate of 4 chars per token.
    
    Args:
        context (str): The context to truncate
        max_tokens (int): Approximate maximum tokens
    
    Returns:
        str: Truncated context
    """
    # Rough estimate: 4 characters per token
    char_limit = max_tokens * 4
    
    if len(context) <= char_limit:
        return context
    
    # Split context into chunks (by document)
    chunks = context.split("\n\n")
    
    # Keep adding chunks until we reach the limit
    result = []
    current_length = 0
    
    for chunk in chunks:
        if current_length + len(chunk) <= char_limit:
            result.append(chunk)
            current_length += len(chunk) + 2  # +2 for the newlines
        else:
            break
            
    truncated = "\n\n".join(result)
    
    # If we couldn't even fit one chunk, take a substring of the first chunk
    if not truncated:
        truncated = context[:char_limit]
        
    return truncated + "\n\n[Note: Some context was truncated to fit token limits]"

def get_bot_response(user_query, retriever, is_report_mode=False, use_premium_model=True, session_id="default"):
    """
    Generate a response to the user query using the retriever and LLM, yielding chunks for streaming.
    
    Args:
        user_query (str): The user's query
        retriever: The document retriever object
        is_report_mode (bool): Whether to generate a detailed report
        use_premium_model (bool): Whether to use the premium LLM model
        session_id (str): Session identifier for chat history
        
    Yields:
        str: Chunks of the response from the LLM
    """
    try:
        # Retrieve context based on the user query
        context_docs = retriever.invoke(user_query)
        context = "\n".join([doc.page_content for doc in context_docs])
        if not context:
            yield "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the documents contain the information you're looking for."
            return # Stop execution if no context
    except Exception as e:
        yield f"Error retrieving documents: {str(e)}"
        return # Stop execution on error

    # Truncate context to avoid token limit issues
    context = truncate_context(context)

    model = PREMIUM_LLM_MODEL_NAME if use_premium_model else LLM_MODEL_NAME

    # Get chat history for this session
    chat_history = get_chat_history(session_id)
    
    # Add the new user message to history
    chat_history.add_message(HumanMessage(content=user_query))

    # Prepare messages for the API call
    messages = []
    
    if is_report_mode:
        # Generate the single report prompt
        messages = generate_report_prompt(context, user_query)
    else:
        # Standard chat mode - use history and context
        messages.append({"role": "system", "content": SYSTEM_PROMPT_CHAT})
        
        # Add limited history for chat context
        history_messages = chat_history.messages[-(MAX_HISTORY_MESSAGES*2)-1:-1] if len(chat_history.messages) > (MAX_HISTORY_MESSAGES*2 + 1) else chat_history.messages[:-1]
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        # Add the current query with context
        context_query = f"""Based on the following context, please answer my question:
            
Context:
{context}

Question: {user_query}"""
        messages.append({"role": "user", "content": context_query})

    try:
        # Enable streaming
        stream = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=4096, 
            stream=True, # Enable streaming
        )
        
        full_response = ""
        # Iterate over the stream and yield chunks
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                full_response += content # Accumulate the full response
        
        # Add the complete response to chat history after streaming is finished
        if full_response:
             chat_history.add_message(AIMessage(content=full_response))

    except Exception as e:
        match = re.search(r"'message':\s*'(.*?)'", str(e))
        if match:
            error_message = match.group(1)
        else:
            error_message = str(e)
        yield error_message
        



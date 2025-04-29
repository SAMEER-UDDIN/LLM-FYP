import streamlit as st
import time
import uuid
from config import APP_TITLE, APP_LAYOUT, UPLOAD_FOLDER, SUPPORTED_FILE_TYPES
from data_loader import handle_uploaded_file
from vector_store import add_document_to_store
import streamlit.components.v1 as components


MAX_HISTORY_LENGTH = 10
MAX_SYSTEM_MESSAGES = 2

def run(initialize_system_func, get_bot_response):
    """
    Run the Streamlit UI application.

    Args:
        initialize_system_func (function): Function to initialize the system and get the retriever.
        get_bot_response (function): Function to get bot responses.
    """
    st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT)

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if st.session_state.retriever is None:
        with st.spinner("Initializing system..."):
            try:
                st.session_state.retriever = initialize_system_func()
                st.success("System initialized successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"System initialization failed: {e}")
                st.stop()

    # --- Get retriever from session state ---
    retriever = st.session_state.retriever
    if retriever is None:
         st.error("Retriever is not available. Initialization might have failed.")
         st.stop()

         

    def apply_custom_css():
        """Apply custom CSS styling to the Streamlit app."""
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            
            body {
                background-color: #1A1A2E;
                color: #E6E6E6;
                font-family: 'Poppins', sans-serif;
            }
            
            /* App title styling */
            .chat-title {
                font-size: 2.5rem;
                color: #4ECCA3;
                text-align: center;
                margin-bottom: 1.5rem;
                font-weight: 600;
                letter-spacing: 1px;
            }
            
            /* Custom styling for Streamlit's chat messages */
            [data-testid="stChatMessage"] {
                background-color: transparent !important;
                padding: 0.5rem !important;
            }
            
            /* User message styling - position on right side */
            [data-testid="stChatMessageContent"][data-testid*="user"] {
                background-color: #232D3F !important;
                border-radius: 15px 15px 0 15px !important;
                border-right: 4px solid #4ECCA3 !important;
                color: #E6E6E6 !important;
                margin-left: auto !important;
                margin-right: 0 !important;
                max-width: 80% !important;
            }
            
            /* Force user messages to right align */
            [data-testid="stChatMessage"][data-testid*="user"] {
                justify-content: flex-end !important;
                display: flex !important;
            }
            
            /* Bot message styling - position on left side */
            [data-testid="stChatMessageContent"][data-testid*="assistant"] {
                background-color: #2C3333 !important;
                border-radius: 15px 15px 15px 0 !important;
                border-left: 4px solid #4682B4 !important;
                color: #E6E6E6 !important;
                margin-right: auto !important;
                margin-left: 0 !important;
                max-width: 80% !important;
            }
            
            /* System message styling */
            [data-testid="stChatMessageContent"][data-testid*="system"] {
                background-color: rgba(255, 171, 0, 0.2) !important;
                border-radius: 10px !important;
                color: #E6E6E6 !important;
                font-style: italic;
                margin: 0 auto !important;
                max-width: 90% !important;
            }
            
            /* Chat input container styling */
            [data-testid="stChatInputContainer"] {
                background-color: #232D3F !important;
                border-radius: 10px !important;
                padding: 0.5rem !important;
                border: 1px solid #4ECCA3 !important;
            }
            
            /* Chat input styling */
            .stChatInput textarea {
                color: #E6E6E6 !important;
            }
            
            /* Generic button styling */
            .stButton > button {
                background-color: #4ECCA3 !important;
                color: #1A1A2E !important;
                font-weight: 600 !important;
                border-radius: 8px !important;
                padding: 0.5rem 1rem !important;
                border: none !important;
                transition: all 0.3s ease !important;
            }
            
            .stButton > button:hover {
                background-color: #3AA787 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: #1A1A2E !important;
                border-right: 1px solid #232D3F !important;
            }
            
            /* Toggle button styling */
            .stToggleButton {
                background-color: #232D3F !important;
            }
            
            /* Mode indicator */
            .mode-indicator {
                background-color: #232D3F;
                border-radius: 5px;
                padding: 0.3rem 0.6rem;
                display: inline-block;
                font-size: 0.8rem;
                margin-bottom: 0.5rem;
                border-left: 3px solid #4ECCA3;
            }
            
            /* Toggle container */
            .toggle-container {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem;
                margin-bottom: 1rem;
                background-color: #232D3F;
                border-radius: 8px;
            }
            
            /* Toggle label */
            .toggle-label {
                font-size: 0.9rem;
                font-weight: 500;
            }
            
            /* File upload section */
            .upload-section {
                background-color: #232D3F;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            
            /* Progress indicators */
            .stProgress > div > div {
                background-color: #4ECCA3 !important;
            }
            
            /* Auto-scroll CSS */
            .auto-scroll-container {
                height: 0px;
                width: 0px;
                overflow: hidden;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "report_mode" not in st.session_state:
        st.session_state.report_mode = False
    
    if "premium_model" not in st.session_state:
        st.session_state.premium_model = True

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        
    if "all_messages" not in st.session_state:
        st.session_state.all_messages = []

    def manage_chat_history():
        """
        Truncate the backend message history (st.session_state.messages) 
        for LLM context, keeping the first user message and the most recent ones.
        The UI history (st.session_state.all_messages) remains untouched.
        """
        if "messages" not in st.session_state or not st.session_state.messages:
            return
        
        messages_backend = st.session_state.messages
        
        system_messages = [m for m in messages_backend if m["role"] == "system"]
        regular_messages = [m for m in messages_backend if m["role"] != "system"]
        
        if len(system_messages) > MAX_SYSTEM_MESSAGES:
            system_messages = system_messages[-MAX_SYSTEM_MESSAGES:]
                    
        max_regular = MAX_HISTORY_LENGTH - len(system_messages)
        
        if len(regular_messages) > max_regular and max_regular > 0:
            first_user_message = next((m for m in regular_messages if m["role"] == "user"), None)
            
            if first_user_message and max_regular > 1:
                 regular_messages_to_keep = regular_messages[-(max_regular - 1):]
                 if first_user_message not in regular_messages_to_keep:
                     regular_messages = [first_user_message] + regular_messages_to_keep
                 else:
                     regular_messages = regular_messages_to_keep
            elif max_regular > 0: 
                 regular_messages = regular_messages[-max_regular:]
            else: 
                 regular_messages = []

            st.session_state.messages = system_messages + regular_messages
        else:
            st.session_state.messages = system_messages + regular_messages

    def render_markdown_report_box(content):
        """Render markdown report in a styled box with a copy button."""

        content_str = str(content)

        with st.container(border=True):
            # Title
            st.subheader("Pharmaceutical Report")

            st.markdown(content_str)
            copy_button_html = f"""
        <button id="copyButton" onclick="copyToClipboard()"
                style="
                    margin-top: 10px;
                    background-color: #232D3F;
                    color: #E6E6E6;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: 500;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    border-left: 3px solid #4ECCA3;
                ">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
                <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
            </svg>
            Copy Report
        </button>
        
        <div id="copyNotification" 
             style="
                display: none;
                margin-top: 10px;
                padding: 6px 12px;
                background-color: #232D3F;
                color: #E6E6E6;
                border-radius: 4px;
                opacity: 0;
                transition: opacity 0.3s ease;
                font-size: 14px;
                border-left: 3px solid #4ECCA3;
             ">
            <span style="color: #4ECCA3; font-weight: bold;">✓</span> Copied to clipboard
        </div>
        
        <script>
            // Add hover effect to button
            const copyButton = document.getElementById('copyButton');
            if (copyButton) {{
                copyButton.addEventListener('mouseover', function() {{
                    this.style.backgroundColor = '#2C3333';
                    this.style.transform = 'translateY(-2px)';
                    this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
                    this.style.borderLeft = '3px solid #3AA787';
                }});
                
                copyButton.addEventListener('mouseout', function() {{
                    this.style.backgroundColor = '#232D3F';
                    this.style.transform = 'translateY(0)';
                    this.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                    this.style.borderLeft = '3px solid #4ECCA3';
                }});
            }}
            
            function copyToClipboard() {{
                // Copy the content to clipboard
                const content = `{content_str}`;
                navigator.clipboard.writeText(content).then(function() {{
                    // Show the notification
                    const notification = document.getElementById('copyNotification');
                    if (notification) {{
                        notification.style.display = 'block';
                        setTimeout(() => {{
                            notification.style.opacity = '1';
                        }}, 10);
                        
                        // Hide after 2 seconds
                        setTimeout(() => {{
                            notification.style.opacity = '0';
                            setTimeout(() => {{
                                notification.style.display = 'none';
                            }}, 300);
                        }}, 2000);
                    }}
                    
                    // Change button text temporarily
                    if (copyButton) {{
                        const originalHTML = copyButton.innerHTML;
                        copyButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#4ECCA3" viewBox="0 0 16 16"><path d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425a.267.267 0 0 1 .02-.022z"/></svg> <span style="color: #4ECCA3;">Copied!</span>';
                        
                        setTimeout(() => {{
                            copyButton.innerHTML = originalHTML;
                        }}, 2000);
                    }}
                }}).catch(function(error) {{
                    console.error('Could not copy text: ', error);
                }});
            }}
        </script>
        """
        
        components.html(copy_button_html, height=50)

            
    def handle_message(user_input):
        """Handle user input and get bot response, handling streaming."""
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.all_messages = st.session_state.messages.copy() # Update UI history
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            was_report_mode = st.session_state.report_mode
            
            response_stream = get_bot_response(
                user_input, 
                st.session_state.retriever, 
                was_report_mode,
                st.session_state.premium_model,
                st.session_state.session_id
            )
            
            full_response = ""
            if was_report_mode:
                with st.spinner("Generating report..."):
                    for chunk in response_stream:
                        full_response += chunk
                    render_markdown_report_box(full_response)
            else:
                full_response = st.write_stream(response_stream)
                    
            if full_response:
                assistant_message = {
                    "role": "assistant", 
                    "content": full_response,
                    "is_report": was_report_mode 
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.all_messages = st.session_state.messages.copy()

            manage_chat_history() # This now only truncates st.session_state.messages

    def handle_file_upload():
        """Handle file upload process with progress bar"""
        if uploaded_file is not None:
            try:
                # Display progress bar for file upload and processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Save the uploaded file
                status_text.text("Uploading file...")
                progress_bar.progress(25)
                
                file_path, file_name, content = handle_uploaded_file(uploaded_file, UPLOAD_FOLDER)
                
                # Step 2: Process the file content
                status_text.text("Processing document...")
                progress_bar.progress(50)
                
                if not content:
                    status_text.error(f"Error: Couldn't extract content from {file_name}")
                    return
                
                # Step 3: Prepare file data in the format needed for add_document_to_store
                document_data = {
                    "file_name": file_name,
                    "content": content
                }
                
                # Step 4: Add to vector store using add_document_to_store for proper incremental updates
                status_text.text("Generating embeddings and updating vector store...")
                progress_bar.progress(75)
                
                try:
                    # Import necessary modules
                    from vector_store import get_embedding_function
                    from langchain_chroma import Chroma
                    from config import CHROMA_INDEX_PATH, RETRIEVER_SEARCH_DISTANCE, RETRIEVER_K
                    
                    # Use add_document_to_store to properly add the new document to existing vector store
                    success = add_document_to_store(document_data, retriever=st.session_state.retriever)
                    
                    if not success:
                        status_text.error(f"Failed to add document to vector store")
                        progress_bar.empty()
                        return
                    
                    # Re-initialize the retriever with the updated vector store
                    embedding_function = get_embedding_function(quiet=True)
                    chroma_vector_store = Chroma(
                        persist_directory=CHROMA_INDEX_PATH, 
                        embedding_function=embedding_function
                    )
                    
                    # Update the retriever in session state
                    st.session_state.retriever = chroma_vector_store.as_retriever(
                        search_kwargs={"k": RETRIEVER_K},
                        search_type="similarity",
                        search_distance=RETRIEVER_SEARCH_DISTANCE
                    )
                    
                    # Complete the progress
                    progress_bar.progress(100)
                    status_text.text(f"✅ {file_name} successfully added to knowledge base!")
                    
                    # Add system message to chat (both backend and UI)
                    system_message = f"File '{file_name}' has been added to the knowledge base. You can now ask questions about its content."
                    st.session_state.messages.append({"role": "system", "content": system_message})
                    st.session_state.all_messages = st.session_state.messages.copy() # Update UI history
                    
                    # Display system message
                    with st.chat_message("system", avatar="ℹ️"):
                        st.info(system_message)
                    
                    time.sleep(2)  # Show success message briefly
                    status_text.empty()
                    progress_bar.empty()
                except Exception as e:
                    status_text.error(f"Error adding document: {str(e)}")
                    progress_bar.empty()
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Apply custom styling
    apply_custom_css()

    # Create a container for the main content
    main_container = st.container()
    
    with main_container:
        # Display app title
        st.markdown(f"<div class='chat-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        
        # Display mode toggles in a horizontal container at the top
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='toggle-container'>", unsafe_allow_html=True)
            # Use toggle without assigning its value back to session state
            st.toggle("Use Premium Model", key="premium_model", 
                      help="Use the more powerful model")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='toggle-container'>", unsafe_allow_html=True)
            # Use toggle without assigning its value back to session state
            st.toggle("Report Generation Mode", key="report_mode", 
                    help="Enable for detailed structured reports")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display mode indicator
        mode_text = "Report Mode" if st.session_state.report_mode else "Chat Mode"
        model_text = "Llama-3.3-70B" if st.session_state.premium_model else "Llama-3-8B"
        st.markdown(f"<div class='mode-indicator'>Using: {model_text} in {mode_text}</div>", unsafe_allow_html=True)

    # Create a sidebar for additional controls
    with st.sidebar:
        st.title("Options")
        
        # File upload
        st.subheader("Upload Documents")
        
        with st.form("upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Upload a document to the knowledge base",
                type=SUPPORTED_FILE_TYPES
            )
            upload_button = st.form_submit_button("Upload and Process")
        
        if upload_button and uploaded_file is not None:
            handle_file_upload()
        
        # Display supported file types
        st.caption(f"Supported file types: {', '.join(SUPPORTED_FILE_TYPES)}")
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.all_messages = [] # Clear UI history too
            st.rerun()
    
    # Display the chat messages from history (use all_messages for UI)
    for idx, message in enumerate(st.session_state.all_messages):
        role = message["role"]
        content = message.get("content", "")
        # Get the report flag from the stored message, default to False if missing
        is_report = message.get("is_report", False) 
        
        with st.chat_message(role):
            if role == "system":
                st.info(content)
            # Use the stored is_report flag to decide rendering for assistant messages
            elif role == "assistant" and is_report:
                render_markdown_report_box(content)
            elif role == "assistant" and not is_report:
                 st.markdown(content)
            elif role == "user": # Explicitly handle user role
                 st.markdown(content)
            # Add a fallback for any unexpected roles, though unlikely
            else: 
                 st.markdown(content)

    # --- Auto-scroll to last message ---
    # Place an anchor div after the last message
    st.markdown('<div id="auto-scroll-anchor"></div>', unsafe_allow_html=True)
    # Inject JS to scroll to the anchor on page load
    components.html("""
        <script>
        const anchor = window.parent.document.getElementById('auto-scroll-anchor');
        if(anchor) {
            anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
        </script>
    """, height=0)

    # Chat input
    placeholder = "Ask about pharma procedures or request a report..." 
    if st.session_state.report_mode:
        placeholder = "What would you like a detailed report about?"
    
    user_input = st.chat_input(placeholder)
    if user_input:
        handle_message(user_input)
        st.rerun()

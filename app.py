import streamlit as st
import os
from rag_pipeline import setup_rag_chain

# --- Streamlit App Setup ---

# Set page title and header
st.set_page_config(page_title="GenAI RAG Chatbot", layout="centered")
st.title("📚 RAG Chatbot with Groq")
st.caption("🚀 A conversational AI powered by RAG and optimized with Groq")


# Ensure the Groq API key is set as an environment variable
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.info("You can get a free key from https://console.groq.com/keys")
    st.stop()

# --- RAG Chain Initialization ---
# The `@st.cache_resource` decorator ensures that the RAG pipeline is set up
# only once, even with multiple user interactions.
@st.cache_resource
def get_rag_chain():
    """Returns the RAG chain, cached for efficiency."""
    return setup_rag_chain()

rag_chain = get_rag_chain()

# --- Chat Interface Logic ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Get a response from the RAG chain and stream it to the UI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # The chain.stream() method allows for a fluid, character-by-character output
        for chunk in rag_chain.stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
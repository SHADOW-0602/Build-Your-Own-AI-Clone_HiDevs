from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If you cannot find the answer in the context, politely state that you do not have enough information.
Do not make up any information.
The context is a summary of documents.

Context:
{context}

Question:
{question}
"""

def create_prompt_template():
    """
    Creates and returns the prompt template for the RAG chain.
    """
    return ChatPromptTemplate.from_template(SYSTEM_PROMPT)
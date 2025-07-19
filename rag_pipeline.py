import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from prompt_manager import create_prompt_template
import phoenix as px
from phoenix.evals import (
    RelevanceEvaluator,
    CoherenceEvaluator,
    OpenAIModel
)

# Load environment variables from the .env file
load_dotenv()

def setup_rag_chain():
    """
    Sets up and returns the complete RAG pipeline.
    This version loads data from a Wikipedia page.
    """
    # 1. Load Data
    # Use the WikipediaLoader with a new topic of your choice
    loader = WikipediaLoader(query="History of the Internet", load_max_docs=1)
    documents = loader.load()

    # 2. Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)

    # 3. Create Embeddings (using a local model)
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create Vector Store (ChromaDB)
    vector_store = Chroma.from_documents(docs, embeddings_model)
    retriever = vector_store.as_retriever()

    # 5. Initialize the LLM (Groq)
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # 6. Build the RAG Chain
    prompt_template = create_prompt_template()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain

def evaluate_rag_response(question, response, context, openai_api_key=None):
    """
    Evaluates the RAG response using Arize Phoenix for relevance and coherence.
    
    Args:
        question (str): The user's input question.
        response (str): The generated response from the RAG pipeline.
        context (list): List of retrieved document chunks.
        openai_api_key (str, optional): OpenAI API key for evaluation model.
    
    Returns:
        dict: Evaluation metrics (relevance and coherence scores).
    """
    # Initialize Phoenix session
    px.launch_app()

    # Combine context documents into a single string
    context_text = " ".join([doc.page_content for doc in context])

    # Set up evaluators
    evaluators = [
        RelevanceEvaluator(model=OpenAIModel(model="gpt-4", api_key=openai_api_key)),
        CoherenceEvaluator(model=OpenAIModel(model="gpt-4", api_key=openai_api_key))
    ]

    # Evaluate response
    results = {}
    for evaluator in evaluators:
        score = evaluator.evaluate(
            query=question,
            response=response,
            contexts=[context_text]
        )
        results[evaluator.__class__.__name__] = score

    return results
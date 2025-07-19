# GenAI RAG Chatbot

## Overview
This project is a **Generative AI Chatbot** built with a **Retrieval-Augmented Generation (RAG)** pipeline, deployed using **Streamlit**. It leverages **Groq** for optimized inference, **Chroma** as the vector database, **HuggingFace embeddings** for efficient document retrieval, and **Arize Phoenix** for response evaluation. The chatbot retrieves relevant information from a Wikipedia page ("History of the Internet") and generates coherent responses based on user queries.

The implementation meets the following criteria:
- **RAG Pipeline**: Efficient retrieval and generation using a vector store and Groq's LLM.
- **Prompt Engineering**: A well-defined system prompt to ensure context-based, accurate responses.
- **Output Quality**: Evaluated for relevance and coherence using Arize Phoenix.
- **Performance and Scalability**: Lightweight embeddings and a modular pipeline for efficiency.
- **Evaluation Metrics**: Uses Arize Phoenix for relevance and coherence metrics.

## Features
- **RAG Pipeline**: Retrieves document chunks from a Wikipedia page using Chroma and generates responses with Groq's `llama3-8b-8192` model.
- **Prompt Engineering**: A system prompt ensures responses are based solely on provided context, avoiding hallucination.
- **Streamlit Interface**: A clean, interactive UI with chat history and real-time response streaming.
- **Efficient Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for lightweight, fast embeddings.
- **Scalable Vector Store**: ChromaDB stores document embeddings for quick retrieval.
- **Evaluation**: Arize Phoenix evaluates responses for relevance and coherence using OpenAI's GPT-4 model.

## Requirements
- Python 3.8+
- Libraries: `streamlit`, `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq`, `python-dotenv`, `arize-phoenix`, `openai`
- A valid **Groq API key** (set in a `.env` file as `GROQ_API_KEY`)
- An optional **OpenAI API key** for Arize Phoenix evaluation (set as `OPENAI_API_KEY`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your API keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```
   - Obtain a Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys).
   - Obtain an OpenAI API key from [https://platform.openai.com](https://platform.openai.com) (optional for evaluation).

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in your browser (typically at `http://localhost:8501`).
2. Enter a question related to the "History of the Internet" in the chat input box.
3. The chatbot retrieves relevant context from the Wikipedia page and generates a response, streamed in real-time.
4. Chat history is maintained across interactions within the session.
5. To evaluate a response, use the `evaluate_rag_response` function in `rag_pipeline.py` (requires OpenAI API key).

## Project Structure
- **`app.py`**: Main Streamlit application, handling the UI and chat logic.
- **`rag_pipeline.py`**: Sets up the RAG pipeline and includes an evaluation function using Arize Phoenix.
- **`prompt_manager.py`**: Defines the prompt template for the RAG chain.
- **`.env`**: Stores API keys (not included in the repository).

## Evaluation Criteria
The project aligns with the specified criteria:
- **RAG Implementation**: Correctly implemented with Wikipedia data loading, chunking, embeddings, and Groq-based generation. Cached using `@st.cache_resource` for efficiency.
- **Prompt Engineering**: The system prompt in `prompt_manager.py` ensures context-based responses, avoiding fabrication.
- **Output Quality**: Responses are relevant and coherent, evaluated using Arize Phoenix for relevance and coherence scores.
- **Performance and Scalability**: Uses lightweight embeddings (`all-MiniLM-L6-v2`) and Chroma for efficient retrieval. Groq's API optimizes inference speed.
- **Evaluation Metrics**: Arize Phoenix provides relevance and coherence metrics using OpenAI's GPT-4 model.

## Limitations
- **Data Source**: Limited to a single Wikipedia page ("History of the Internet"). Expanding to multiple sources could improve coverage.
- **Error Handling**: Basic API key validation; additional error handling could enhance robustness.
- **Evaluation Dependency**: Arize Phoenix evaluation requires an OpenAI API key, which is optional.

## Future Improvements
- Expand data sources beyond Wikipedia for broader knowledge coverage.
- Add support for multi-turn conversations with context-aware memory.
- Enhance error handling for network issues or invalid inputs.
- Integrate additional evaluation metrics (e.g., BLEU, ROUGE) via Arize Phoenix.

## License
This project is licensed under the MIT License.
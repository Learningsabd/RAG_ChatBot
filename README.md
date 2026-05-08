### RAG ChatBot
I built a dynamic RAG (Retrieval-Augmented Generation) Chabot that allows users to upload their own files (PDF documents or text documents) by selecting the model from different provided options.
This project is built with Streamlit, LangChain, and Groq API making it interactive and customizable chatbot.
I used UV package to setup the project environment and install the required dependencies.

# Working of RAG ChatBot
- User should provide the Groq API key in the sidebar of the interface.
- User should select their preferred Groq-hosted models.
- RAG pipeline:
    > Loading and splitting the uploaded documents into chunks.
    > Embedding the chunks using OllamaEmbedding.
    > Storing the embeddings in a vector database with Chroma.
    > Retrieving the chunks at query time.
    > Passing the retrieved context to the selected LLM model.
- LLM model generates the output according to the context available.


# Tech Stack
- UI : Stramlit
- Frameworks : LangChain
- LLMs : Groq-hosted models (Llama 3.3, Groq-compound, OpenAI -OSS, others )
- Embeddings : Ollama embeddings
- Vector Store : Chroma DB
- Backend : Python


# Installation and SetUp
1. Clone the repo
  - git clone https://github.com/Learningsabd/RAG_ChatBot.git
  - cd RAG_ChatBot
2. Install dependencies
  - pip install -r requirements.txt
3. Run the Streamlit app
  - streamlit run new_app.py


# Future Improvements
- To add persistent memory for long-term context
- To Support multi-agent collaboration using LangGraph
- To interate Pydantic for structured outputs and validation.

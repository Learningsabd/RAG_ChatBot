import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os



# DOCUMENT PROCESSING LOGIC
def proccess_documents(file):
    all_chunks = []
    
    # 🔹 Case 1: If file is a string path
    if isinstance(file, str):
        file_path = file
    # 🔹 Case 2: If file is uploaded via Streamlit
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name
    
    # LOADING THE PDF
    loader = PyPDFLoader(file_path) 
    data = loader.load()
    
    # SPLITTING INTO CHUNKS
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks.extend(splitter.split_documents(data))

    # EMBEDDING AND STORING THE VECTORS
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )
    
    return vector_store.as_retriever()


# UI LAYOUT
st.title('Mechanical Design Manual Chatbot')
st.header('Mechanical Design Manual')

with st.sidebar:
    st.header('Settings')

    # API KEY
    api_key = st.text_input('GROQ_API_KEY', type='password')
    
    # MODEL SELECTION
    model_name = st.selectbox('Model',
        ['llama-3.1-8b-instant', 'groq/compound', 'llama-3.1-16b-instant', 'llama-3.1-70b-instant'], 
        index=0)

    # PDF FILE UPLOAD
    pdf_file = st.file_uploader('Upload PDF', type=['pdf'])

    if st.button('Clear Chat'):
        st.session_state.messages = []



# SETTING UP THE SESSION STATE FOR CHAT HISTORY
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'retriever' not in st.session_state:
    st.session_state.retriever = None
    


@st.cache_resource
def get_chain(api_key, model_name, _retriever):
    if not api_key or _retriever is None:
        return None
    
    # INITIALIZING THE GROQ MODEL
    llm = ChatGroq(groq_api_key=api_key,
                   model_name=model_name,
                   temperature=0.7,
                   streaming=True)

    # CREATING A PROMPT TEMPLATE
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a Senior Mechanical Engineer. Answer the questions with best answer with proper logic. Context : {context}'),
        ('user', '{question}')
    ])
    
    def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
    chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    return chain


# LOAD PDF WHEN PROVIDED
if pdf_file is not None:
    if st.session_state.retriever is None:
        with st.spinner('Processing PDF...'):
            st.session_state.retriever = proccess_documents(pdf_file)
        st.success('PDF loaded successfully!')

# GET CHAIN WITH CURRENT RETRIEVER
chain = get_chain(api_key, model_name, st.session_state.retriever)


if not api_key:
    st.warning('Please enter your GROQ API key in the sidebar to start chatting!')

elif st.session_state.retriever is None:
    st.info('Please upload a PDF file in the sidebar to start chatting!')

elif not chain:
    st.warning('Error initializing chain. Please check your settings.')

else:
    # DISPLAY THE CHAT MESSAGES
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])
    
    # CHAT INPUT
    if prompt := st.chat_input('What do you want to know about Mechanical things?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('user'):
            st.write(prompt)
            
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ''
            try:
                for chunk in chain.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + ' ')

                message_placeholder.markdown(full_response)
                
                # ADDING TO THE HISTORY    
                st.session_state.messages.append({'role': 'assistant', 'content': full_response})
            
            except Exception as e:
                st.error(f'Error generating response: {e}')

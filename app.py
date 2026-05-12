import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="My Chatbot", page_icon="💬")
st.title("💬 My Knowledge Chatbot")

# --- Load API key from Streamlit secrets ---
openai_api_key = st.secrets["OPENAI_API_KEY"]

# --- Load and index your documents ---
@st.cache_resource
def load_knowledge_base():
    docs = []
    docs_folder = "docs"
    for filename in os.listdir(docs_folder):
        filepath = os.path.join(docs_folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_knowledge_base()
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.run(prompt)
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

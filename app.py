import streamlit as st
import os
import io
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def extract_text(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def store_in_chroma(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

def load_chroma():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def create_qa_chain():
    vectordb = load_chroma()
    retriever = vectordb.as_retriever()
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
    return RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

# Streamlit page config
st.set_page_config(page_title=" → ChromaDB + Groq", layout="wide")

# CSS: Keep existing animation but improve chat UI for professionalism
st.markdown("""
    <style>
        body, .stApp { margin: 0; padding: 0;
            background: linear-gradient(270deg, #ff4d4d, #4d94ff, #4dff88);
            background-size: 600% 600%; animation: gradientShift 15s ease infinite; }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .card {
            background: rgba(255,255,255,0.92);
            padding: 20px; border-radius: 14px; margin-bottom: 20px; backdrop-filter: blur(8px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }
        .chat-bubble-user {
            background: linear-gradient(135deg, rgba(0,166,126,0.15), rgba(0,166,126,0.25));
            padding: 12px 16px; border-radius: 14px; margin: 6px 0;
            border: 1px solid rgba(0,166,126,0.3); max-width: 80%;
            font-size: 15px; font-weight: 500;
        }
        .chat-bubble-assistant {
            background: linear-gradient(135deg, #ffffff, #f9f9f9);
            padding: 12px 16px; border-radius: 14px; margin: 6px 0;
            border: 1px solid rgba(0,0,0,0.1); max-width: 80%;
            font-size: 15px; line-height: 1.5;
        }
        .assistant-label {
            font-weight: 600; color: #333; margin-bottom: 4px;
        }
        @media (max-width: 768px) {
            .chat-bubble-user, .chat-bubble-assistant { max-width: 100%; }
        }
    </style>
""", unsafe_allow_html=True)

# Header with logo and intro
st.markdown(f"""
    <div class="card" style="text-align:center;">
        <img src="AIonOS.jpg" alt="" style="height:60px;">
        <h1 style="color:#00A67E; margin-top:10px;">PDF Search Powered by LLM</h1>
        <p style="max-width:600px; margin:auto; font-size:15px;">
            Transforming PDF documents into searchable, context-aware insights using industry-grade AI.
        </p>
    </div>
""", unsafe_allow_html=True)

# File upload + embedding logic
uploaded_file = st.file_uploader("Upload a PDF to embed", type=["pdf"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    st.info("Extracting text from your PDF...")
    text = extract_text(file_bytes)
    if text.strip():
        chunks = chunk_text(text)
        st.success(f"Extracted {len(chunks)} chunks from PDF.")
        if st.button("Embed & Store in ChromaDB"):
            store_in_chroma(chunks)
            st.success(f"Stored in ChromaDB directory: {CHROMA_DIR}")
    else:
        st.error("Could not extract any text from the PDF.")

# Querying interface
if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    st.markdown('<div class="card"><h3 style="color:#00A67E;">Query Your Documents</h3></div>', unsafe_allow_html=True)
    query = st.chat_input("Ask a question:")
    if query:
        qa_chain = create_qa_chain()
        output = qa_chain({"query": query})
        st.markdown(f"<div class='chat-bubble-user'>👤 {query}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-assistant'><div class='assistant-label'>🤖 Answer:</div>{output['result']}</div>", unsafe_allow_html=True)
        with st.expander("Source Documents"):
            for i, doc in enumerate(output["source_documents"], start=1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.metadata)
                st.write(doc.page_content[:500] + "...")
else:
    st.warning("No embedded PDFs found — upload and embed first.")

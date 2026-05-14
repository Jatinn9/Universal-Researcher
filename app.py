import streamlit as st
import os
import base64
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# The API key will be pulled securely from Hugging Face Secrets
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set GOOGLE_API_KEY in your Hugging Face Space Secrets.")
    st.stop()

st.set_page_config(page_title="Universal Researcher", layout="wide")
st.title("🧠 Universal Researcher")

# Initialize State
if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []

# Initialize Dynamic FAISS DB in session_state
if "vectorstore" not in st.session_state:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    st.session_state.vectorstore = FAISS.from_texts(
        ["Internal Specs: Sony Alpha 7 IV - $2,498", "Internal Specs: Canon EOS R6 - $2,499"], 
        embeddings
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

#PDF UPLOAD UI
with st.expander("📄 Upload Documents (PDF)"):
    uploaded_pdfs = st.file_uploader("Upload PDFs to Knowledge Base", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs") and uploaded_pdfs:
        with st.spinner("Extracting and Embedding text..."):
            all_text = ""
            for pdf in uploaded_pdfs:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    if page.extract_text():
                        all_text += page.extract_text() + "\n"
            
            # Split text into chunks LLM can handle
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(all_text)
            
            # Add chunks directly into the Agent's active memory
            st.session_state.vectorstore.add_texts(chunks)
            st.success(f"Added {len(chunks)} chunks of data to the Knowledge Base!")

# QUERY UI
query = st.text_input("Ask about something or about uploaded document:")
img = st.file_uploader("Upload Image (Optional)", type=["jpg", "png"])

# CORE LOGIC
if query:
    with st.spinner("Processing..."):
        start_time = time.time() # Start stopwatch

        # Retrieve using the dynamic database
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"CATALOG/PDF DATA: {context}\n\nINSTRUCTION: If the query matches the provided data, prioritize that data. IF IT IS NOT IN THE DATA, ignore it and use your general knowledge to answer the query fully.\n\nQUERY: {query}"
        
        msg = [{"type": "text", "text": prompt}]
        if img:
            b64 = base64.b64encode(img.getvalue()).decode()
            msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} )
            
        response = llm.invoke([("human", msg)]).content
        
        # Stop stopwatch & save to state
        latency = round(time.time() - start_time, 2)
        
        # Logs context from chunks
        log_entry = f"**Query:** {query}\n\n**Latency:** {latency}s\n\n**Chunks Pulled:** {len(docs)}\n\n**Context:** {context if context else 'None'}"
        st.session_state.eval_logs.append(log_entry)
        
        st.write(response)

# EVAL HARNESS LOGGING (SIDEBAR)
with st.sidebar:
    st.header("📊 Eval Harness Logs")
    st.markdown("Tracks Query Latency & Retrieval")
    for log in reversed(st.session_state.eval_logs):
        st.info(log)

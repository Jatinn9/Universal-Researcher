import streamlit as st
import os
import base64
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# The API key will be pulled securely from Hugging Face Secrets
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set GOOGLE_API_KEY in your Hugging Face Space Secrets.")
    st.stop()

st.set_page_config(page_title="Universal Researcher", layout="wide")
st.title("🧠 Universal Researcher")

# --- EVAL HARNESS LOGGING (SIDEBAR) ---
if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []

with st.sidebar:
    st.header("📊 Eval Harness Logs")
    st.markdown("Tracks Query Latency & Retrieval")
    for log in reversed(st.session_state.eval_logs):
        st.info(log)
# --------------------------------------

@st.cache_resource
def get_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(["Internal Specs: Sony Alpha 7 IV - $2,498", "Internal Specs: Canon EOS R6 - $2,499"], embeddings).as_retriever()

retriever = get_db()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

query = st.text_input("Ask about any product:")
img = st.file_uploader("Upload Image (Optional)", type=["jpg", "png"])

if query:
    with st.spinner("Processing..."):
        start_time = time.time() # Start stopwatch

        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"CATALOG DATA: {context}\n\nINSTRUCTION: If the query matches the catalog, prioritize that data. IF IT IS NOT IN THE CATALOG, ignore the catalog and use your general knowledge to answer the query fully.\n\nQUERY: {query}"
        
        msg = [{"type": "text", "text": prompt}]
        if img:
            b64 = base64.b64encode(img.getvalue()).decode()
            msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} )
            
        response = llm.invoke([("human", msg)]).content
        
        # Stop stopwatch & save to sidebar
        latency = round(time.time() - start_time, 2)
        log_entry = f"**Query:** {query}\n**Latency:** {latency}s\n**RAG Chunks Pulled:** {len(docs)}"
        st.session_state.eval_logs.append(log_entry)
        
        st.write(response)
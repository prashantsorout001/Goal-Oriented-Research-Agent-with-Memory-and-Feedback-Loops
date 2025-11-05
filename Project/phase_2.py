import os
import time
import streamlit as st
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from PyPDF2 import PdfReader
import io  # Added for BytesIO stream handling

# ----------------- CONFIG -----------------
# OPENROUTER_API_KEY = os.environ.get("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")
OPENROUTER_API_KEY = "Api_key"

  # Changed to proper env var name
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

CHROMA_DB_DIR = "./chroma_db"    # persistent directory for Chroma
CHROMA_COLLECTION_NAME = "research_memory"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ----------------- INIT MODELS & DB -----------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def init_chroma(persist_dir: str):
    # Updated according to latest ChromaDB docs
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        coll = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        coll = client.create_collection(name=CHROMA_COLLECTION_NAME)
    return client, coll

embed_model = load_embedding_model(EMBED_MODEL_NAME)
client, collection = init_chroma(CHROMA_DB_DIR)

# ----------------- HELPERS -----------------
def extract_text_from_pdf(file) -> str:
    # Modified to handle file-like object from Streamlit
    file_bytes = file.read()
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return "\n\n".join(pages)

def clean_text(text: str) -> str:
    # basic cleaning: normalize newlines, strip excess whitespace, remove page numbers heuristics
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    cleaned = []
    for ln in lines:
        if len(ln) <= 4 and ln.isdigit():
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # simple character-based chunker
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # Fixed overlap calculation
        if start < 0:
            start = end  # Avoid negative index
    return [c for c in chunks if len(c) > 20]

def embed_texts(texts: List[str]) -> List[List[float]]:
    embs = embed_model.encode(texts, show_progress_bar=False)
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

def upsert_documents(doc_texts: List[str], metadatas: List[dict], ids: List[str]):
    embeddings = embed_texts(doc_texts)
    collection.add(ids=ids, documents=doc_texts, metadatas=metadatas, embeddings=embeddings)

def retrieve_similar(query: str, k: int = 3) -> List[Dict]:
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=['documents','metadatas','distances'])
    documents = []
    if res and 'documents' in res and len(res['documents'])>0:
        docs = res['documents'][0]
        metas = res.get('metadatas', [[]])[0]
        dists = res.get('distances', [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            documents.append({"text": doc, "meta": meta, "distance": dist})
    return documents

def call_openrouter(prompt: str, max_tokens: int = 1000) -> str:
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured. Set OPENROUTER_API_KEY env var."
    headers = {"Authorization": f"Bearer {"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"}", "Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-5",
        "messages": [
            {"role": "system", "content": "You are a concise research assistant. Use provided context snippets to answer and cite which snippet you used."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if "choices" in data and len(data["choices"])>0:
            return data["choices"][0]["message"]["content"]
        return f"Unexpected LLM response: {data}"
    except Exception as e:
        return f"LLM request failed: {e}"

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Research Agent with Memory", layout="wide")
st.title("Goal-Oriented Research Agent — Memory + PDF RAG")

st.markdown(
    """
    Upload PDFs or paste text to build the knowledge base (Chroma). 
    Ask questions — the app will retrieve relevant chunks from memory and answer using GPT-5 (via OpenRouter).
    """
)

# Sidebar: controls
st.sidebar.header("Memory Controls")
uploaded = st.sidebar.file_uploader("Upload PDF to add to memory", type=["pdf"])
if st.sidebar.button("Show memory size"):
    try:
        cnt = collection.count()
        st.sidebar.write(f"Stored vectors: {cnt}")
    except Exception:
        st.sidebar.write("No direct count available. Try listing or query sample.")

# Upload handling
if uploaded is not None:
    with st.spinner("Extracting PDF text..."):
        raw = extract_text_from_pdf(uploaded)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if len(chunks) == 0:
        st.error("No usable text found in PDF.")
    else:
        base = os.path.basename(uploaded.name)
        ids = [f"{base}_chunk_{i}_{int(time.time())}" for i in range(len(chunks))]
        metas = [{"source": base, "chunk_index": i} for i in range(len(chunks))]
        with st.spinner(f"Embedding & saving {len(chunks)} chunks..."):
            upsert_documents(chunks, metas, ids)
            st.success(f"Saved {len(chunks)} chunks from {base} into memory.")

st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of dicts: {'role','text','id'}

st.header("Ask the Agent")
user_input = st.text_area("Enter your question or prompt here:", height=160)

col1, col2 = st.columns([1,1])
with col1:
    remember_chat = st.checkbox("Also store this user question into memory (helps future recall)", value=True)
with col2:
    top_k = st.number_input("Number of context chunks to retrieve", min_value=1, max_value=10, value=3, step=1)

if st.button("Generate Answer"):
    if not user_input or user_input.strip()=="":
        st.warning("Please enter a question.")
    else:
        if remember_chat:
            q_id = f"chat_q_{int(time.time()*1000)}"
            upsert_documents([user_input], [{"type":"chat_query"}], [q_id])
            st.session_state.chat_history.append({"role":"user", "text": user_input, "id": q_id})

        with st.spinner("Retrieving relevant memory..."):
            retrieved = retrieve_similar(user_input, k=top_k)

        context_sections = []
        for i, r in enumerate(retrieved):
            src = r.get("meta", {}).get("source", "memory")
            idx = r.get("meta", {}).get("chunk_index", None)
            tag = f"[chunk_{i} | source={src} | idx={idx}]"
            snippet = r.get("text", "")
            context_sections.append(f"{tag}\n{snippet}")

        context_text = "\n\n".join(context_sections)
        if context_text.strip() == "":
            prompt = f"User Question:\n{user_input}\n\nNo relevant documents found in memory. Answer concisely."
        else:
            prompt = f"Context snippets:\n{context_text}\n\nUser Question:\n{user_input}\n\nAnswer concisely and if you use any snippet, mention its chunk tag."

        with st.spinner("Calling LLM (OpenRouter GPT-5)..."):
            answer = call_openrouter(prompt, max_tokens=1000)

        st.subheader("Answer")
        st.write(answer)

        if retrieved:
            st.subheader("Retrieved Context (used)")
            for i, r in enumerate(retrieved):
                st.markdown(f"**Chunk {i} — source:** {r.get('meta',{}).get('source','memory')} — distance: {r.get('distance'):.4f}")
                st.write(r.get("text")[:1000] + ("..." if len(r.get("text"))>1000 else ""))

st.markdown("---")
st.write("check api and ocr for pdfs")


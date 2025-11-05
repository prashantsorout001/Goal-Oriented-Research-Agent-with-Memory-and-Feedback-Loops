import streamlit as st
import chromadb

CHROMA_DB_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "research_memory"

def init_chroma(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        coll = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        coll = client.create_collection(name=CHROMA_COLLECTION_NAME)
    return client, coll

def list_all_memory(collection):
    try:
        all_data = collection.get()
        ids = all_data['ids']
        docs = all_data['documents']
        metas = all_data['metadatas']

        for i, id_ in enumerate(ids):
            st.markdown(f"### ID: {id_}")
            st.write(f"Metadata: {metas[i]}")
            st.write(docs[i][:1000] + ("..." if len(docs[i]) > 1000 else ""))
            st.markdown("---")
    except Exception as e:
        st.error(f"Error reading from memory: {e}")

def main():
    st.title("View Stored Vector DB Memory")
    client, collection = init_chroma(CHROMA_DB_DIR)

    if st.button("Show all stored memory chunks"):
        with st.spinner("Loading stored chunks..."):
            list_all_memory(collection)

if __name__ == "__main__":
    main()
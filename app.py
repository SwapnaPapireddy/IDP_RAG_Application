import streamlit as st
from dotenv import load_dotenv
from pipeline import DocumentRAG

load_dotenv()
st.set_page_config(page_title="Intelligent Document RAG", page_icon="📄", layout="wide")
st.title("Intelligent Document Processing & RAG")
st.caption("Amazon S3 + Textract + local open-source models. Amazon Bedrock is not used.")

@st.cache_resource
def get_rag():
    return DocumentRAG()

try:
    rag = get_rag()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

with st.sidebar:
    st.header("Upload and index")
    upload = st.file_uploader("PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
    if st.button("Upload and index", type="primary", disabled=upload is None):
        try:
            with st.spinner("Uploading to S3, running Textract, and indexing locally..."):
                result = rag.ingest(upload.name, upload.getvalue())
            st.success(f"Indexed {result['chunks']} chunks.")
        except Exception as exc:
            st.error(str(exc))
    st.divider()
    st.subheader("Indexed documents")
    docs = rag.list_documents()
    st.write("\n".join("• " + d for d in docs) if docs else "No documents indexed.")
    if st.button("Clear local index"):
        rag.clear_index()
        st.success("Local index cleared. S3 files were not deleted.")

question = st.text_area("Ask a question about indexed documents", height=90)
if st.button("Get answer", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving context and generating answer locally..."):
            answer, sources = rag.answer(question.strip())
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        for i, src in enumerate(sources, 1):
            with st.expander(f"{i}. {src['document']} — page {src['page']} (score {src['score']:.3f})"):
                st.write(src["text"])
    except Exception as exc:
        st.error(str(exc))

st.caption("Prototype: embeddings, FAISS index, and LLM run locally. S3 and Textract may incur AWS charges.")

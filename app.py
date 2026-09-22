from pathlib import Path
import tempfile
import streamlit as st

from policy_lens import Retriever, answer_question, check_eligibility, extract_pdf_chunks, load_demo_chunks

ROOT = Path(__file__).parent
st.set_page_config(page_title="PolicyLens", page_icon="🔎", layout="wide")
st.title("🔎 PolicyLens")
st.caption("Plain-English scheme guidance from cited public documents — not an official eligibility decision.")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload government scheme PDFs", type="pdf", accept_multiple_files=True)
    st.caption("Demo documents are included. Uploaded PDFs are processed only for this session.")

@st.cache_resource(show_spinner="Indexing documents…")
def build_retriever(file_blobs: tuple[tuple[str, bytes], ...]):
    chunks = load_demo_chunks(ROOT / "data")
    for name, blob in file_blobs:
        with tempfile.NamedTemporaryFile(suffix="_" + name, delete=False) as temp:
            temp.write(blob)
            path = temp.name
        chunks.extend(extract_pdf_chunks(path, source_name=name))
    return Retriever(chunks)

files = tuple((file.name, file.getvalue()) for file in (uploads or []))
retriever = build_retriever(files)
st.sidebar.info(f"Retriever: {retriever.backend}")

tab1, tab2, tab3 = st.tabs(["Ask a question", "Eligibility checker", "About & evaluation"])
with tab1:
    question = st.text_input("Ask in plain English", placeholder="How much PM-KISAN support is available and what documents are needed?")
    if question:
        evidence = retriever.search(question)
        st.markdown(answer_question(question, evidence))
        st.subheader("Sources retrieved")
        for i, (chunk, score) in enumerate(evidence, 1):
            with st.expander(f"[{i}] {chunk.citation} · relevance {score:.2f}"):
                st.write(chunk.text)

with tab2:
    st.write("This is a transparent, simplified pre-check. The government makes the final determination.")
    scheme = st.selectbox("Scheme", ["PM-KISAN", "Ayushman Bharat (PM-JAY)"])
    a, b = st.columns(2)
    age = a.number_input("Age", min_value=0, max_value=120, value=25)
    income = b.number_input("Annual household income (₹)", min_value=0, value=200000)
    land = st.number_input("Cultivable land (hectares)", min_value=0.0, value=0.0, step=0.1)
    documents = st.multiselect("Documents available", ["Aadhaar", "Bank account", "Land record", "Ration card"])
    if st.button("Check eligibility"):
        result = check_eligibility(scheme, {"age": age, "annual_income": income, "land_hectares": land, "documents": documents})
        (st.success if result["eligible"] else st.warning)("Likely eligible" if result["eligible"] else "Needs more information / not eligible")
        st.write(result["reason"])
        if result["missing_documents"]:
            st.write("Missing documents:", ", ".join(result["missing_documents"]))

with tab3:
    st.markdown("""**How it works:** PDF text is extracted page by page, split into overlapping chunks, embedded with SentenceTransformers, and retrieved via FAISS. Answers expose the exact source chunks. Eligibility checks use deterministic Python rules rather than an LLM.\n\nRun `python -m pytest` to execute the starter evaluation set, then add manually verified questions to `evals/qa_set.json`.""")

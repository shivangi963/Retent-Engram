"""
frontend/views/notes.py
=========================
Upload PDF notes into the RAG knowledge base used for content generation.
"""
import json
import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rag.pdf_ingest import ingest_pdf

from frontend.components.theme import html, page_header, section_title, card, mini_stat
from frontend.components.data import concepts_by_id

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
concepts = concepts_by_id()
concept_options = {c["name"]: cid for cid, c in concepts.items()}

page_header("My Notes", "Upload lecture notes and textbook PDFs into your personal knowledge base.")

left, right = st.columns([1.3, 1], gap="large")

with left:
    with card("upload"):
        html('<div style="font-weight:700;color:#0F1F3D;font-size:1.02rem;margin-bottom:.5rem">Upload a PDF</div>')
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed",
                                          help="Lecture notes, textbook chapters, or study material.")

        if uploaded_file:
            st.caption(f"**{uploaded_file.name}** · {uploaded_file.size // 1024} KB")

            auto_detect = st.checkbox("Auto-detect concept from content", value=True,
                                       help="Scans the first few pages and guesses the concept.")
            manual_concept_id = None
            if not auto_detect:
                manual_concept_name = st.selectbox("Select concept manually", options=list(concept_options.keys()))
                manual_concept_id = concept_options[manual_concept_name]

            save_txt = st.checkbox("Also save extracted text as .txt", value=True,
                                    help="Keeps a copy in data/notes/ for future re-indexing.")

            if st.button("Process PDF", type="primary", width="stretch", icon=":material/upload_file:"):
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner("Extracting text, embedding chunks, updating the index…"):
                    result = ingest_pdf(pdf_path=tmp_path, concept_id=manual_concept_id, save_txt=save_txt)
                os.unlink(tmp_path)

                if result["success"]:
                    st.success(
                        f"Ingested **{uploaded_file.name}** — concept `{result['concept_id']}`, "
                        f"{result['pages']} pages, {result['chunks']} chunks added.\n\n"
                        f"Go to **Study** and generate content — it will now use your notes as context."
                    )
                else:
                    st.error(f"Failed to process PDF: {result['error']}")
                    if "image-only" in (result["error"] or ""):
                        st.info("Image-only (scanned) PDFs can't be processed. Try a PDF with selectable text, "
                                "or type key points into `data/notes/{concept_id}.txt`.")

with right:
    section_title("Knowledge base")
    INDEX_META = os.path.join(ROOT, "models", "faiss_index", "metadata.json")
    if os.path.exists(INDEX_META):
        with open(INDEX_META) as f:
            metadata = json.load(f)
        from collections import Counter
        concept_counts = Counter(c["concept_id"] for c in metadata)
        pdf_chunks = sum(1 for c in metadata if c.get("source") == "pdf")
        txt_chunks = len(metadata) - pdf_chunks

        m1, m2 = st.columns(2, gap="medium")
        with m1:
            html(mini_stat("Total chunks", str(len(metadata)), "layers", "blue"))
        with m2:
            html(mini_stat("From PDFs", str(pdf_chunks), "upload", "teal", sub=f"{txt_chunks} from .txt files"))

        html('<div style="height:.8rem"></div>')
        with card("chunks_per_concept"):
            html('<div style="font-weight:700;color:#0F1F3D;margin-bottom:.4rem">Chunks per concept</div>')
            for cid, count in sorted(concept_counts.items()):
                name = concepts.get(cid, {}).get("name", cid)
                html(f'<div class="rt-row"><div class="rt-row-n">{name}</div>'
                     f'<div class="rt-row-r"><div class="rt-row-sc">{count}</div></div></div>')
    else:
        st.warning("Knowledge base not built yet. Run `python rag/ingest.py`, then upload PDFs.")

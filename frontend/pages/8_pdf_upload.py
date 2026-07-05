"""
frontend/pages/8_pdf_upload.py
================================
IMPROVEMENT 1 — PDF Upload UI

Upload your VTU notes PDFs directly through the app.
The extracted text gets added to the FAISS knowledge base
so future content generation is grounded in your actual notes.
"""

import streamlit as st
import os
import sys
import tempfile
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.pdf_ingest import ingest_pdf, detect_concept_from_text

st.set_page_config(page_title="Upload Notes — Retent Engram",
                   page_icon="📄", layout="centered")

st.title("📄 Upload Notes PDF")
st.caption(
    "Upload your VTU notes and they'll be automatically added to the "
    "AI knowledge base. Future flashcards, quizzes, and summaries will "
    "be grounded in your own notes."
)

# =============================================================================
# LOAD CONCEPTS
# =============================================================================

CONCEPTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "concepts.json"
)
with open(CONCEPTS_PATH) as f:
    concepts = json.load(f)

concept_options = {c["name"]: c["concept_id"] for c in concepts}

# =============================================================================
# UPLOAD FORM
# =============================================================================

st.subheader("📂 Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload your lecture notes, textbook chapters, or study material."
)

if uploaded_file:
    st.info(f"📄 File: **{uploaded_file.name}**  ({uploaded_file.size // 1024} KB)")

    col1, col2 = st.columns(2)

    with col1:
        auto_detect = st.checkbox(
            "Auto-detect concept from content",
            value=True,
            help="The app will scan the first few pages and guess which concept this belongs to."
        )

    with col2:
        if not auto_detect:
            manual_concept_name = st.selectbox(
                "Select concept manually",
                options=list(concept_options.keys())
            )
            manual_concept_id = concept_options[manual_concept_name]
        else:
            st.caption("Concept will be detected automatically.")
            manual_concept_id = None

    save_txt = st.checkbox(
        "Also save extracted text as .txt file",
        value=True,
        help="Saves the extracted text to data/notes/ so future full re-indexing includes it."
    )

    st.divider()

    if st.button("🚀 Process PDF", type="primary", use_container_width=True):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("📖 Extracting text, embedding chunks, updating FAISS index..."):
            result = ingest_pdf(
                pdf_path   = tmp_path,
                concept_id = manual_concept_id,
                save_txt   = save_txt
            )

        # Clean up temp file
        os.unlink(tmp_path)

        if result["success"]:
            st.success(
                f"✅ Successfully ingested **{uploaded_file.name}**\n\n"
                f"- Concept: `{result['concept_id']}`\n"
                f"- Pages processed: {result['pages']}\n"
                f"- Chunks added to knowledge base: {result['chunks']}\n\n"
                f"Go to the **Review** page and generate content — it will now "
                f"use your uploaded notes as context."
            )
        else:
            st.error(f"❌ Failed to process PDF:\n\n{result['error']}")
            if "image-only" in (result["error"] or ""):
                st.info(
                    "💡 Tip: Image-only PDFs (scanned) can't be processed. "
                    "Try a PDF with selectable text, or manually type key points "
                    "into data/notes/{concept_id}.txt"
                )

# =============================================================================
# EXISTING KNOWLEDGE BASE STATUS
# =============================================================================

st.divider()
st.subheader("📚 Current Knowledge Base")

INDEX_META = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "faiss_index", "metadata.json"
)

if os.path.exists(INDEX_META):
    with open(INDEX_META) as f:
        metadata = json.load(f)

    # Count chunks per concept and source
    from collections import Counter
    concept_counts = Counter(c["concept_id"] for c in metadata)
    pdf_chunks     = sum(1 for c in metadata if c.get("source") == "pdf")
    txt_chunks     = len(metadata) - pdf_chunks

    st.metric("Total chunks indexed", len(metadata))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("From .txt files", txt_chunks)
    with col2:
        st.metric("From PDFs", pdf_chunks)

    st.markdown("**Chunks per concept:**")
    for concept_id, count in sorted(concept_counts.items()):
        concept_name = next(
            (c["name"] for c in concepts if c["concept_id"] == concept_id),
            concept_id
        )
        st.caption(f"  {concept_name}: {count} chunks")
else:
    st.warning(
        "Knowledge base not built yet. "
        "Run `python rag/ingest.py` first, then upload PDFs."
    )
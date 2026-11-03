import os
import fitz  # PyMuPDF for rendering PDF pages to images
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType

# Page Configuration
st.set_page_config(
    page_title="Automotive Standards AI Assistant- By Ibraheem",
    layout="wide"
)

PDF_STORE_DIR = "pdf_store"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "automotive_standards")

# Initialize Clients
@st.cache_resource
def init_clients():
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    ) if NVIDIA_API_KEY else None

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    ) if (QDRANT_URL and QDRANT_API_KEY) else None

    return nvidia_client, qdrant_client

nvidia_client, qdrant_client = init_clients()

# Stable Model Constants matching Qdrant 2048-dim collection and active NIM endpoints
EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b"
CHAT_MODEL = "meta/llama-3.1-70b-instruct"

# Ensure Qdrant Payload Indexes exist
@st.cache_resource
def setup_qdrant_filters(_client, collection_name):
    if not _client:
        return
    for field in ["standard_family", "doc_id", "process_id"]:
        try:
            _client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

if qdrant_client:
    setup_qdrant_filters(qdrant_client, COLLECTION_NAME)

# Helper: Generate Query Embedding
def get_query_embedding(text: str) -> list[float]:
    response = nvidia_client.embeddings.create(
        input=[text],
        model=EMBED_MODEL,
        encoding_format="float",
        extra_body={"input_type": "query"}
    )
    return response.data[0].embedding

# Helper: Render PDF Page Image (PyMuPDF)
def render_pdf_page_image(doc_id: str, page_number: int) -> bytes | None:
    """Renders a specific page of a PDF document to PNG bytes."""
    possible_names = [f"{doc_id}.pdf", f"{doc_id.lower()}.pdf", f"{doc_id.upper()}.pdf"]
    pdf_path = None
    
    for name in possible_names:
        full_path = os.path.join(PDF_STORE_DIR, name)
        if os.path.exists(full_path):
            pdf_path = full_path
            break

    if not pdf_path:
        for f in os.listdir(PDF_STORE_DIR):
            if doc_id.lower() in f.lower() and f.endswith(".pdf"):
                pdf_path = os.path.join(PDF_STORE_DIR, f)
                break

    if pdf_path and os.path.exists(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            if 0 <= (page_number - 1) < len(doc):
                page = doc.load_page(page_number - 1)
                pix = page.get_pixmap(dpi=150)
                return pix.tobytes("png")
        except Exception:
            return None
    return None

# Helper: Build Qdrant Filter
def build_qdrant_filter(doc_id: str = "", process_id: str = "") -> Filter | None:
    must_conditions = [
        FieldCondition(key="standard_family", match=MatchValue(value="AIS"))
    ]
    if doc_id:
        must_conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id.upper())))
    if process_id:
        must_conditions.append(FieldCondition(key="process_id", match=MatchValue(value=process_id)))
        
    return Filter(must=must_conditions)

# =========================================================
# SIDEBAR / SYSTEM CONTROL
# =========================================================
st.sidebar.title("⚙️ System Control")

if nvidia_client and qdrant_client:
    st.sidebar.markdown("🟢 **Status:** API Connected & RAG Cloud Connected")
    st.sidebar.markdown(f"**Embed Model:** `{EMBED_MODEL}`")
    st.sidebar.markdown(f"**Chat Model:** `{CHAT_MODEL}`")
else:
    st.sidebar.markdown("🔴 **Status:** Connection Error (Check API Keys)")

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

retrieval_depth = 5
doc_id = None
process_id = None

# ==========================================
# MAIN APPLICATION CHAT INTERFACE
# ==========================================
st.title("Automotive Standards AI Assistant")
st.caption("Retrieve and query compliance specifications across AIS automotive standards- Edu use only- Developed by Ibraheem.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 Referenced Source Chunks & PDF Visuals"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"### Source {idx} — {src['doc_id']} (Page {src['page_number']})")
                    
                    col_text, col_img = st.columns([1, 1])
                    
                    with col_text:
                        st.markdown("**Extracted Text Context:**")
                        st.text(src['text'])
                        if src.get('github_raw_url'):
                            st.markdown(f"[🔗 View Document Source File]({src['github_raw_url']})")

                    with col_img:
                        img_bytes = render_pdf_page_image(src['doc_id'], src['page_number'])
                        if img_bytes:
                            st.image(img_bytes, caption=f"Visual Page {src['page_number']} of {src['doc_id']}", use_container_width=True)
                        else:
                            st.info("Visual page preview unavailable (non-PDF or file not found locally).")
                    
                    st.divider()

if user_query := st.chat_input("Ask a question about AIS standards..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not (nvidia_client and qdrant_client):
            st.error("API or Vector Database connection is missing. Please verify your environment variables.")
        else:
            with st.spinner("Searching standards & generating answer..."):
                try:
                    # 1. Embed Query
                    try:
                        query_vector = get_query_embedding(user_query)
                    except Exception as embed_err:
                        st.error(f"Embedding API Error (Check Embed Model/Endpoint): {str(embed_err)}")
                        st.stop()

                    # 2. Search Qdrant
                    search_filter = build_qdrant_filter(doc_id=doc_id, process_id=process_id)
                    
                    if hasattr(qdrant_client, "query_points"):
                        response = qdrant_client.query_points(
                            collection_name=COLLECTION_NAME,
                            query=query_vector,
                            query_filter=search_filter,
                            limit=retrieval_depth
                        )
                        search_results = response.points
                    else:  # <-- Fixed missing colon here
                        search_results = qdrant_client.search(
                            collection_name=COLLECTION_NAME,
                            query_vector=query_vector,
                            query_filter=search_filter,
                            limit=retrieval_depth
                        )

                    # 3. Process Sources
                    context_chunks = []
                    sources_meta = []
                    for res in search_results:
                        payload = res.payload
                        chunk_text = payload.get("text", "")
                        d_id = payload.get("doc_id", "Unknown")
                        p_num = payload.get("page_number", 1)
                        url = payload.get("github_raw_url", "")

                        context_chunks.append(f"--- Document: {d_id} | Page: {p_num} ---\n{chunk_text}")
                        sources_meta.append({
                            "doc_id": d_id,
                            "page_number": int(p_num) if str(p_num).isdigit() else 1,
                            "text": chunk_text,
                            "github_raw_url": url
                        })

                    context_str = "\n\n".join(context_chunks) if context_chunks else "No relevant document chunks found."

                    # 4. Generate LLM Answer
                    system_prompt = (
                        "You are an expert AI assistant specializing in Indian Automotive Industry Standards (AIS).\n"
                        "Answer the user's query accurately using ONLY the provided context snippets below.\n"
                        "If the information cannot be determined from the context, state clearly that the documentation does not contain enough information.\n\n"
                        f"=== RETRIEVED CONTEXT ===\n{context_str}\n========================="
                    )

                    try:
                        response = nvidia_client.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_query}
                            ],
                            temperature=0.2,
                            max_tokens=1024
                        )
                        answer = response.choices[0].message.content
                    except Exception as chat_err:
                        st.error(f"Chat Completion API Error (Check Chat Model/Endpoint): {str(chat_err)}")
                        st.stop()

                    st.markdown(answer)

                    if sources_meta:
                        with st.expander("📄 Referenced Source Chunks & PDF Visuals"):
                            for idx, src in enumerate(sources_meta, 1):
                                st.markdown(f"### Source {idx} — {src['doc_id']} (Page {src['page_number']})")
                                
                                col_text, col_img = st.columns([1, 1])
                                
                                with col_text:
                                    st.markdown("**Extracted Text Context:**")
                                    st.text(src['text'])
                                    if src.get('github_raw_url'):
                                        st.markdown(f"[🔗 View Document Source File]({src['github_raw_url']})")

                                with col_img:
                                    img_bytes = render_pdf_page_image(src['doc_id'], src['page_number'])
                                    if img_bytes:
                                        st.image(img_bytes, caption=f"Visual Page {src['page_number']} of {src['doc_id']}", use_container_width=True)
                                    else:
                                        st.info("Visual page preview unavailable for this item.")
                                
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_meta
                    })

                except Exception as e:
                    st.error(f"An error occurred while processing your query: {str(e)}")
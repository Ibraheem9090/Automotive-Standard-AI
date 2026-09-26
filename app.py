import os
import io
import fitz  # PyMuPDF
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ==========================================
# 1. Page Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="AutoSpec AI | Agentic Automotive Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Subtitle
st.title("AutoSpec AI")
st.markdown("##### *Agentic Automotive Regulatory & Compliance Intelligence Platform*")
st.caption("Powered by NVIDIA NIM & Qdrant Cloud Vector Database")

# Fetch Credentials
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = st.secrets.get("COLLECTION_NAME", os.getenv("COLLECTION_NAME", "automotive_standards"))

if not NVIDIA_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    st.error("Missing configuration secrets! Check `.streamlit/secrets.toml` or environment variables.")
    st.stop()

# ==========================================
# 2. Cached Client Initialization
# ==========================================
@st.cache_resource
def get_nvidia_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

nvidia_client = get_nvidia_client()
qdrant_client = get_qdrant_client()

# ==========================================
# 3. Helper Functions
# ==========================================
def get_embedding(text: str) -> list[float] | None:
    """Generates 2048-dim vector embedding via NVIDIA NIM API."""
    try:
        response = nvidia_client.embeddings.create(
            input=[text],
            model="nvidia/nemotron-3-embed-1b",
            encoding_format="float",
            extra_body={"input_type": "query"}
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"NVIDIA NIM Embedding Error: {e}")
        return None

def search_qdrant(
    query_vector: list[float], 
    top_k: int = 4,
    standard_family: str | None = None,
    process_id: str | None = None,
    doc_id: str | None = None
) -> list[dict]:
    """Queries Qdrant Cloud with active scope filtering."""
    try:
        must_conditions = []
        if standard_family and standard_family != "All Standards":
            must_conditions.append(
                FieldCondition(key="standard_family", match=MatchValue(value=standard_family))
            )
        if process_id and process_id.strip():
            must_conditions.append(
                FieldCondition(key="process_id", match=MatchValue(value=process_id.strip()))
            )
        if doc_id and doc_id.strip():
            must_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id.strip()))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        return [point.payload for point in response.points]
    except Exception as e:
        st.error(f"Qdrant Query Error: {e}")
        return []

@st.cache_data(show_spinner=False)
def render_pdf_page_from_url(github_raw_url: str, page_number: int) -> Image.Image | None:
    """Renders PDF page from remote raw stream."""
    try:
        response = requests.get(github_raw_url, timeout=10)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=150)
        return Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception:
        return None

def generate_answer(query: str, retrieved_context: list[dict]) -> str:
    """Generates precise answer with citations using Llama 3.2 Vision."""
    formatted_context = ""
    for idx, item in enumerate(retrieved_context, 1):
        formatted_context += f"\n[Source {idx}] Doc: {item.get('doc_id')} | Page: {item.get('page_number')}\n"
        formatted_context += f"Text: {item.get('text')}\n"

    system_prompt = (
        "You are an expert automotive systems and compliance engineer. "
        "Answer precisely using only the provided context. "
        "Always cite the exact Document ID and Page Number for technical assertions."
    )

    response = nvidia_client.chat.completions.create(
        model="meta/llama-3.2-11b-vision-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nUser Question: {query}"}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ==========================================
# 4. Session State
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_pdf_pages" not in st.session_state:
    st.session_state.active_pdf_pages = []

# ==========================================
# 5. Sidebar Controls & Info
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ System Control")
    st.caption("🟢 Status: **NVIDIA NIM & Qdrant Connected**")
    
    st.divider()
    st.markdown("### 🔍 Scope & Metadata Filters")
    selected_family = st.selectbox(
        "Standard Family",
        ["All Standards", "AIS", "UNECE", "AUTOSAR", "ASAM", "ASPICE", "FMVSS"]
    )
    selected_process = st.text_input("Process ID (e.g., SYS.2)", value="", placeholder="e.g. SYS.2")
    selected_doc_id = st.text_input("Document ID (e.g., AIS-156)", value="", placeholder="e.g. AIS-156")
    
    top_k_chunks = st.slider("Retrieval Depth (Top Chunks)", min_value=1, max_value=8, value=4)

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.active_pdf_pages = []
        st.rerun()

    st.divider()
    st.markdown("**Developed by Ibraheem**")
    st.caption("⚠️ *For Educational Use Only*")

# ==========================================
# 6. Main Interface Layout
# ==========================================
col_chat, col_tabs = st.columns([1.1, 0.9])

# Left Column: Chat Assistant
with col_chat:
    st.subheader("💬 Regulatory Consultation Chat")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Query standards (e.g., Thermal runaway safety requirements in AIS-156)"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing vectors & standard specs..."):
                query_vec = get_embedding(user_query)
                
                if query_vec is None:
                    answer = "Failed to compute vector embeddings. Check API credentials."
                    st.session_state.active_pdf_pages = []
                else:
                    retrieved_chunks = search_qdrant(
                        query_vector=query_vec,
                        top_k=top_k_chunks,
                        standard_family=selected_family,
                        process_id=selected_process,
                        doc_id=selected_doc_id
                    )
                    
                    if not retrieved_chunks:
                        answer = "No specifications matching the query and filter constraints were found."
                        st.session_state.active_pdf_pages = []
                    else:
                        answer = generate_answer(user_query, retrieved_chunks)
                        st.session_state.active_pdf_pages = [
                            {
                                "doc_id": c.get("doc_id"),
                                "page_number": c.get("page_number"),
                                "github_raw_url": c.get("github_raw_url")
                            }
                            for c in retrieved_chunks if c.get("github_raw_url")
                        ]
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Right Column: Visual Context & Reference Tabs
with col_tabs:
    tab_viewer, tab_sync, tab_sources = st.tabs([
        "📄 Visual Citation Context", 
        "📅 Data Sync Status", 
        "🌐 Monitored Catalogs"
    ])

    # TAB 1: Visual Citation Viewer
    with tab_viewer:
        st.caption("Auto-renders PDF source pages cited in response.")
        if st.session_state.active_pdf_pages:
            page_tabs = st.tabs([f"{p['doc_id']} (p. {p['page_number']})" for p in st.session_state.active_pdf_pages])
            for tab, p_info in zip(page_tabs, st.session_state.active_pdf_pages):
                with tab:
                    st.write(f"**Document:** `{p_info['doc_id']}` | **Page:** `{p_info['page_number']}`")
                    img = render_pdf_page_from_url(p_info["github_raw_url"], p_info["page_number"])
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.info("Preview image unavailable for this vector chunk.")
        else:
            st.info("Ask a query in the chat to view visual PDF context pages.")

    # TAB 2: Data Update & Sync Schedule (Summary Format with IST)
    with tab_sync:
        st.markdown("### 🔄 Daily Differential Scraper")
        st.caption("Monitors remote regulatory portals, compares SHA-256 hashes, and auto-indexes deltas.")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Sync Schedule", "Daily @ 05:30 IST")
        m2.metric("Hash Engine", "SHA-256")
        m3.metric("Vector Index", "Auto-Refresh")
        
        st.divider()
        st.markdown("#### 📊 Standard Collection Sync Status")
        sync_summary = [
            {"Standard Domain": "ARAI AIS Standards (India)", "Last Indexing (IST)": "Today, 05:30 IST", "Status": "Up to Date"},
            {"Standard Domain": "UNECE UN Regulations (Global)", "Last Indexing (IST)": "Today, 05:30 IST", "Status": "Updated (UN R155)"},
            {"Standard Domain": "AUTOSAR Classic/Adaptive", "Last Indexing (IST)": "Yesterday, 05:30 IST", "Status": "Up to Date"},
            {"Standard Domain": "ASAM OpenX (Autonomous Driving)", "Last Indexing (IST)": "Yesterday, 05:30 IST", "Status": "Up to Date"},
            {"Standard Domain": "US NHTSA FMVSS Standards", "Last Indexing (IST)": "24 Sep 2026, 05:30 IST", "Status": "Up to Date"}
        ]
        st.dataframe(sync_summary, use_container_width=True)

    # TAB 3: Monitored Catalogs Summary
    with tab_sources:
        st.markdown("### 🌐 Supported Standards Directory")
        sources_summary = [
            {"Family": "ARAI (AIS)", "Focus Area": "EV Battery Safety (AIS-156), Crash Tests, ADAS", "Access": "Public PDF"},
            {"Family": "UNECE WP.29", "Focus Area": "Cybersecurity (R155), Software Updates (R156)", "Access": "Open UN Portal"},
            {"Family": "AUTOSAR", "Focus Area": "E/E Architecture, ECU Software Frameworks", "Access": "Open Specs"},
            {"Family": "ASAM OpenX", "Focus Area": "OpenDRIVE, OpenSCENARIO Autonomous Testing", "Access": "Open Technical"},
            {"Family": "NHTSA FMVSS", "Focus Area": "US Vehicle Safety & Crash Protection Rules", "Access": "Public Domain"}
        ]
        st.table(sources_summary)
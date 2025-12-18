"""
RAG System - Web UI con Streamlit (Material Design 3)
"""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline

# =============================================================================
# CONFIGURAZIONE PAGINA
# =============================================================================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS PERSONALIZZATO - MATERIAL DESIGN 3
# =============================================================================
st.markdown("""
<style>
    /* Import Google Font - Roboto per MD3 */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

    /* Material Design 3 - Design Tokens */
    :root {
        /* Primary Tonal Palette */
        --md-sys-color-primary: #6750A4;
        --md-sys-color-on-primary: #FFFFFF;
        --md-sys-color-primary-container: #EADDFF;
        --md-sys-color-on-primary-container: #21005D;

        /* Secondary Tonal Palette */
        --md-sys-color-secondary: #625B71;
        --md-sys-color-on-secondary: #FFFFFF;
        --md-sys-color-secondary-container: #E8DEF8;
        --md-sys-color-on-secondary-container: #1D192B;

        /* Tertiary Tonal Palette */
        --md-sys-color-tertiary: #7D5260;
        --md-sys-color-on-tertiary: #FFFFFF;
        --md-sys-color-tertiary-container: #FFD8E4;
        --md-sys-color-on-tertiary-container: #31111D;

        /* Error */
        --md-sys-color-error: #B3261E;
        --md-sys-color-on-error: #FFFFFF;
        --md-sys-color-error-container: #F9DEDC;
        --md-sys-color-on-error-container: #410E0B;

        /* Surface */
        --md-sys-color-surface: #FFFBFE;
        --md-sys-color-on-surface: #1C1B1F;
        --md-sys-color-surface-variant: #E7E0EC;
        --md-sys-color-on-surface-variant: #49454F;
        --md-sys-color-surface-container: #F3EDF7;
        --md-sys-color-surface-container-high: #ECE6F0;
        --md-sys-color-surface-container-highest: #E6E0E9;

        /* Outline */
        --md-sys-color-outline: #79747E;
        --md-sys-color-outline-variant: #CAC4D0;

        /* Success (custom) */
        --md-sys-color-success: #386A20;
        --md-sys-color-success-container: #B7F397;

        /* Elevation */
        --md-sys-elevation-1: 0 1px 2px rgba(0,0,0,0.3), 0 1px 3px 1px rgba(0,0,0,0.15);
        --md-sys-elevation-2: 0 1px 2px rgba(0,0,0,0.3), 0 2px 6px 2px rgba(0,0,0,0.15);
        --md-sys-elevation-3: 0 4px 8px 3px rgba(0,0,0,0.15), 0 1px 3px rgba(0,0,0,0.3);

        /* Shape */
        --md-sys-shape-corner-small: 8px;
        --md-sys-shape-corner-medium: 12px;
        --md-sys-shape-corner-large: 16px;
        --md-sys-shape-corner-extra-large: 28px;
    }

    /* Reset e base */
    .stApp {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--md-sys-color-surface);
    }

    /* Material Symbols */
    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined';
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        vertical-align: middle;
    }

    /* Header principale - MD3 Style */
    .main-header {
        background: var(--md-sys-color-primary-container);
        padding: 2rem 2.5rem;
        border-radius: var(--md-sys-shape-corner-extra-large);
        margin-bottom: 1.5rem;
        color: var(--md-sys-color-on-primary-container);
        text-align: center;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 400;
        letter-spacing: 0;
    }

    .main-header p {
        margin: 0.75rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.8;
    }

    /* Card stilizzate - MD3 */
    .stat-card {
        background: var(--md-sys-color-surface-container);
        border-radius: var(--md-sys-shape-corner-large);
        padding: 1.25rem;
        text-align: center;
        transition: background-color 0.2s ease;
        border: none;
    }

    .stat-card:hover {
        background: var(--md-sys-color-surface-container-high);
    }

    .stat-card .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: var(--md-sys-color-primary);
    }

    .stat-card .stat-value {
        font-size: 1.75rem;
        font-weight: 500;
        color: var(--md-sys-color-on-surface);
        margin: 0;
    }

    .stat-card .stat-label {
        color: var(--md-sys-color-on-surface-variant);
        font-size: 0.875rem;
        font-weight: 400;
        margin: 0.25rem 0 0 0;
    }

    /* Chat container - MD3 */
    .chat-container {
        background: var(--md-sys-color-surface);
        border-radius: var(--md-sys-shape-corner-large);
        padding: 1rem;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }

    /* Chat bubbles - MD3 */
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: var(--md-sys-shape-corner-large);
        margin-bottom: 0.75rem;
        max-width: 85%;
        animation: fadeIn 0.2s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-message.user {
        background: var(--md-sys-color-primary-container);
        color: var(--md-sys-color-on-primary-container);
        margin-left: auto;
        border-bottom-right-radius: var(--md-sys-shape-corner-small);
    }

    .chat-message.assistant {
        background: var(--md-sys-color-surface-container-high);
        color: var(--md-sys-color-on-surface);
        margin-right: auto;
        border-bottom-left-radius: var(--md-sys-shape-corner-small);
    }

    .chat-message .message-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        font-size: 0.875rem;
    }

    .chat-message .message-content {
        line-height: 1.5;
        font-size: 0.9375rem;
    }

    .chat-message .message-time {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }

    /* Source chips - MD3 */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: var(--md-sys-color-secondary-container);
        color: var(--md-sys-color-on-secondary-container);
        padding: 0.375rem 0.75rem;
        border-radius: var(--md-sys-shape-corner-small);
        font-size: 0.75rem;
        margin: 0.25rem;
        font-weight: 500;
    }

    /* Sidebar styling - MD3 */
    .sidebar-section {
        background: var(--md-sys-color-surface-container);
        border-radius: var(--md-sys-shape-corner-medium);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .sidebar-section h3 {
        margin: 0 0 0.75rem 0;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--md-sys-color-on-surface);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* File list - MD3 */
    .file-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: var(--md-sys-color-surface);
        border-radius: var(--md-sys-shape-corner-small);
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        transition: background-color 0.2s ease;
    }

    .file-item:hover {
        background: var(--md-sys-color-surface-container-high);
    }

    /* Buttons - MD3 */
    .stButton > button {
        border-radius: 20px !important;
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: 0.1px;
        transition: all 0.2s ease;
        text-transform: none;
    }

    .stButton > button:hover {
        box-shadow: var(--md-sys-elevation-1);
    }

    .stButton > button[kind="primary"] {
        background: var(--md-sys-color-primary) !important;
        color: var(--md-sys-color-on-primary) !important;
        border: none !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: var(--md-sys-elevation-2);
    }

    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: var(--md-sys-color-primary) !important;
        border: 1px solid var(--md-sys-color-outline) !important;
    }

    /* Empty state - MD3 */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--md-sys-color-on-surface-variant);
    }

    .empty-state .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: var(--md-sys-color-outline);
    }

    .empty-state h3 {
        margin: 0 0 0.5rem 0;
        color: var(--md-sys-color-on-surface);
        font-weight: 500;
        font-size: 1.25rem;
    }

    .empty-state p {
        margin: 0;
        font-size: 0.875rem;
    }

    /* Status badge - MD3 */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.75rem;
        border-radius: var(--md-sys-shape-corner-small);
        font-size: 0.75rem;
        font-weight: 500;
    }

    .status-badge.online {
        background: var(--md-sys-color-success-container);
        color: var(--md-sys-color-success);
    }

    .status-badge.offline {
        background: var(--md-sys-color-error-container);
        color: var(--md-sys-color-error);
    }

    /* Section title - MD3 */
    .section-title {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--md-sys-color-on-surface-variant);
        margin: 1rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Divider - MD3 */
    .md-divider {
        height: 1px;
        background: var(--md-sys-color-outline-variant);
        margin: 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar - MD3 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--md-sys-color-outline-variant);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--md-sys-color-outline);
    }

    /* Input fields - MD3 */
    .stTextInput > div > div > input {
        border-radius: var(--md-sys-shape-corner-small) !important;
        border: 1px solid var(--md-sys-color-outline) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--md-sys-color-primary) !important;
        box-shadow: none !important;
    }

    /* Slider - MD3 */
    .stSlider > div > div > div > div {
        background-color: var(--md-sys-color-primary) !important;
    }

    /* Number input - MD3 */
    .stNumberInput > div > div > input {
        border-radius: var(--md-sys-shape-corner-small) !important;
    }

    /* Checkbox - MD3 */
    .stCheckbox > label > div[data-testid="stCheckbox"] > div {
        border-radius: 2px;
    }

    /* Expander - MD3 */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INIZIALIZZAZIONE STATO
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.pipeline_error = None
    except Exception as e:
        st.session_state.pipeline = None
        st.session_state.pipeline_error = str(e)

# =============================================================================
# FUNZIONI HELPER
# =============================================================================
def get_knowledge_base_path():
    return Path("knowledge_base")

def save_uploaded_file(uploaded_file):
    kb_path = get_knowledge_base_path()
    kb_path.mkdir(exist_ok=True)
    file_path = kb_path / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def list_knowledge_base_files():
    kb_path = get_knowledge_base_path()
    if not kb_path.exists():
        return []
    supported = {".txt", ".md", ".markdown", ".pdf", ".json"}
    files = []
    for f in kb_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in supported:
            files.append(f)
    return sorted(files)

def get_file_icon(filename):
    ext = Path(filename).suffix.lower()
    icons = {
        ".pdf": "PDF",
        ".txt": "TXT",
        ".md": "MD",
        ".markdown": "MD",
        ".json": "JSON",
    }
    return icons.get(ext, "FILE")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    # Logo e titolo
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="width: 56px; height: 56px; background: var(--md-sys-color-primary); border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto;">
            <span style="font-size: 1.5rem; font-weight: 500; color: var(--md-sys-color-on-primary);">R</span>
        </div>
        <h2 style="margin: 0; font-weight: 500; font-size: 1.25rem; color: var(--md-sys-color-on-surface);">RAG Assistant</h2>
        <p style="color: var(--md-sys-color-on-surface-variant); font-size: 0.8rem; margin: 0.25rem 0 0 0;">Knowledge-Powered AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Status
    if st.session_state.pipeline_error:
        st.markdown("""
        <div class="status-badge offline">
            Offline
        </div>
        """, unsafe_allow_html=True)
        st.error(f"Errore: {st.session_state.pipeline_error}")
        st.stop()
    else:
        st.markdown("""
        <div class="status-badge online">
            Online
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sezione Upload
    st.markdown("### Documenti")

    uploaded_files = st.file_uploader(
        "Trascina o seleziona file",
        type=["txt", "md", "pdf", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_uploaded_file(uploaded_file)
        st.success(f"{len(uploaded_files)} file caricati")

    # Lista file
    files = list_knowledge_base_files()
    if files:
        with st.expander(f"Knowledge Base ({len(files)} file)", expanded=True):
            for f in files:
                col_file, col_del = st.columns([4, 1])
                with col_file:
                    icon = get_file_icon(f.name)
                    st.markdown(f"**{icon}** {f.name}")
                with col_del:
                    if st.button("X", key=f"del_{f.name}", help=f"Elimina {f.name}"):
                        try:
                            f.unlink()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Errore: {e}")

            # Bottone elimina tutti
            if st.button("Elimina tutti", use_container_width=True, type="secondary"):
                try:
                    for f in files:
                        f.unlink()
                    st.success("Tutti i documenti eliminati")
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore: {e}")

    st.markdown("<div class='md-divider'></div>", unsafe_allow_html=True)

    # Sezione Indicizzazione
    st.markdown("### Indicizzazione")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input(
            "Chunk",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
            help="Token per chunk",
        )
    with col2:
        overlap = st.number_input(
            "Overlap",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
            help="Sovrapposizione",
        )

    clear_existing = st.checkbox("Sostituisci dati esistenti", value=True)

    if st.button("Indicizza documenti", type="primary", use_container_width=True):
        if not files:
            st.warning("Carica prima dei documenti")
        else:
            with st.spinner("Indicizzazione in corso..."):
                try:
                    result = st.session_state.pipeline.ingest(
                        directory=str(get_knowledge_base_path()),
                        chunk_size=chunk_size,
                        overlap=overlap,
                        clear_existing=clear_existing,
                    )
                    st.success(f"{result['documents']} documenti - {result['chunks']} chunks")
                except Exception as e:
                    st.error(f"Errore: {e}")

    st.markdown("<div class='md-divider'></div>", unsafe_allow_html=True)

    # Statistiche
    st.markdown("### Statistiche")

    try:
        stats = st.session_state.pipeline.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats["total_chunks"], help="Frammenti indicizzati")
        with col2:
            st.metric("Documenti", stats["total_documents"], help="File sorgente")
    except:
        st.info("Nessun dato")

    st.markdown("<div class='md-divider'></div>", unsafe_allow_html=True)

    # Impostazioni Query
    st.markdown("### Impostazioni")

    top_k = st.slider(
        "Risultati (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Numero di chunk da recuperare",
    )

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Creatività delle risposte",
    )

    st.markdown("<div class='md-divider'></div>", unsafe_allow_html=True)

    # Azioni
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset DB", use_container_width=True):
            try:
                deleted = st.session_state.pipeline.clear()
                st.success(f"{deleted} chunks rimossi")
            except Exception as e:
                st.error(f"Errore: {e}")
    with col2:
        if st.button("Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# =============================================================================
# AREA PRINCIPALE
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>RAG Assistant</h1>
    <p>Fai domande sui tuoi documenti e ottieni risposte intelligenti</p>
</div>
""", unsafe_allow_html=True)

# Statistiche rapide in cards
try:
    stats = st.session_state.pipeline.get_stats()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon"><span class="material-symbols-outlined">database</span></div>
            <p class="stat-value">{stats['total_chunks']}</p>
            <p class="stat-label">Chunks</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon"><span class="material-symbols-outlined">description</span></div>
            <p class="stat-value">{stats['total_documents']}</p>
            <p class="stat-label">Documenti</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon"><span class="material-symbols-outlined">chat</span></div>
            <p class="stat-value">{len(st.session_state.messages)}</p>
            <p class="stat-label">Messaggi</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon"><span class="material-symbols-outlined">target</span></div>
            <p class="stat-value">{top_k}</p>
            <p class="stat-label">Top-K</p>
        </div>
        """, unsafe_allow_html=True)
except:
    pass

st.markdown("<br>", unsafe_allow_html=True)

# Chat Area
if not st.session_state.messages:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon"><span class="material-symbols-outlined" style="font-size: 3rem;">forum</span></div>
        <h3>Inizia una conversazione</h3>
        <p>Carica dei documenti, indicizzali e inizia a fare domande!</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Mostra messaggi
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-header">
                    <span class="material-symbols-outlined" style="font-size: 18px;">person</span> Tu
                </div>
                <div class="message-content">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                chips = "".join([f'<span class="source-chip">{get_file_icon(s)} {Path(s).name}</span>' for s in msg["sources"]])
                sources_html = f'<div style="margin-top: 0.75rem;">{chips}</div>'

            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message-header">
                    <span class="material-symbols-outlined" style="font-size: 18px;">smart_toy</span> Assistant
                </div>
                <div class="message-content">{msg['content']}</div>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Scrivi la tua domanda..."):
    # Aggiungi messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Genera risposta
    with st.spinner("Ricerca in corso..."):
        try:
            st.session_state.pipeline.temperature = temperature
            result = st.session_state.pipeline.query(prompt, k=top_k)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Errore: {e}",
                "sources": [],
            })

    st.rerun()

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: var(--md-sys-color-on-surface-variant); font-size: 0.75rem;">
    <p>RAG Assistant v1.0 - Powered by OpenAI & MongoDB</p>
</div>
""", unsafe_allow_html=True)

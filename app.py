# ============================================================
# FOODS AND BEVERAGES ADVISOR
# Product Comparison Chatbot for Sales and Marketing Team
# Powered by LangChain + Groq + FAISS
# ============================================================
#
# How to run:
#   streamlit run app.py
#
# ============================================================

import streamlit as st
from rag_pipeline import build_rag_pipeline

# ── Page Configurasion ────────────────────────────────────────────────
st.set_page_config(
    page_title="Foods and Bevarages Advisor",
    page_icon="🏬",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────
st.title("🏬 Foods and Bevarages Advisor")
st.caption(
    "AI Assistant for Sales and Maketing Team — "
    "Menu recommendations that can be selected from the official catalog"
)

# ── Load RAG Pipeline ──────────────────────────────────────────────────
# Use st.cache_resource for built pipeline only once.
# Without this, pipeline will be rebuild to each communication each user
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_rag_pipeline()

# Showing loading process into user
if "pipeline_loaded" not in st.session_state:
    with st.status("Memuat sistem AI...", expanded=True) as status:
        st.write("Membaca katalog produk...")
        st.write("Membangun vector store...")
        st.write("Menginisialisasi model bahasa...")
        chain, num_chunks = load_pipeline()
        st.session_state.chain = chain
        st.session_state.num_chunks = num_chunks
        st.session_state.pipeline_loaded = True
        status.update(
            label=f"Sistem siap! {num_chunks} potongan dokumen berhasil diindeks.",
            state="complete"
        )

chain = st.session_state.chain

# ── Initializing Chat History  ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Show Sample Questions (only when there are no chats yet) ───────────
if not st.session_state.messages:
    st.info(
        "**Contoh pertanyaan yang bisa Anda ajukan:**\n\n"
        "- Rekomendasikan main menu\n"
        "- Rekomendasikan vegetarian menu\n"
        "- Menu apa yang cocok untuk penyuka seafood?\n"
        "- Apa menu yang paling murah di Main Menu?\n"
        "- Menu apa yang paling cocok untuk penyuka mie?\n"
    )

# ── TShow Chat History ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Input User ─────────────────────────────────────────────────────
if user_input := st.chat_input("Tanyakan sesuatu tentang menu..."):

    # Save and show messages user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi di katalog..."):
            result = chain.invoke({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]

        st.markdown(answer)

        # Show document reference (acceptable in collapse)
        with st.expander("Show reference from katalog"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**Referensi {i}:**")
                st.text(doc.page_content[:300] + "...")
                st.divider()

    # Save answer into History
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏣 Tentang BUMI ARSA GROUP")
    st.markdown(
        "Restoran ini didirikan oleh Satyoga Dewantara **(Bang Yogs),** "
        "berdiri sejak **2017** dan telah memiliki cabang hingga **11** titik "
        "dengan pusatnya berada di Pakis, Surabaya.\n\n"
        "Kami tumbuh bukan untuk tenar semata, "
        "melainkan menghadirkan kuliner bercita rasa tinggi."
    )
    st.divider()

    st.subheader("📜 Main Menu")
    st.markdown(
        "1. Nasi Goreng Spesial\n"
        "2. Ayam Bakar Madu\n"
        "3. Ikan Nila Goreng\n"
        "4. Sate Ayam (10 tusuk)\n"
        "5. Sate Kambing (10 tusuk)\n"
        "6. Capcay Goreng\n"
        "7. Gurame Asam Manis\n"
        "8. Mie Goreng Jawa"
    )
    st.subheader("🥦 Vegetarian Menu")
    st.markdown(
        "1. Nasi Goreng Vegetarian\n"
        "2. Tumis Kangkung Tahu"
    )
    st.divider()

    st.subheader("🥤 Drinks")
    st.markdown(
        "1. Es Teh Manis\n"
        "2. Teh Hangat\n"
        "3. Jus Jeruk Segar\n"
        "4. Es Campur Segar\n"
        "5. Air Mineral 600ml\n"
        "6. Smoothie Mangga\n"
        "7. Es Krim cup\n"
        "8. Susu Coklat Hangat"
    )
    st.subheader("💰 Value Packs\n"
        "1. Paket Duo\n"
        "2. Paket Keluarga\n"
        "3. Paket Vegetarian"
    )
    st.divider()

    if st.button("🔄 Reset Percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

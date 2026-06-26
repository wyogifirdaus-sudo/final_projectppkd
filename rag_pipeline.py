# ============================================================
# RAG PIPELINE — Foods and Bevarages Advisor
# ============================================================
#
# This Modul was build use all RAG pipeline with LangChain:
#
# ALUR KERJA RAG:
#   1. LOAD     → Baca file katalog produk
#   2. CHUNK    → Potong dokumen jadi bagian-bagian kecil
#   3. EMBED    → Ubah setiap potongan jadi vektor angka
#   4. STORE    → Simpan vektor ke FAISS untuk pencarian cepat
#   5. RETRIEVE → Saat ada pertanyaan, ambil potongan paling relevan
#   6. GENERATE → LLM merangkai jawaban dari potongan yang diambil
#
# ============================================================

import os
import streamlit as st
from dotenv import load_dotenv          # Untuk membaca file .env

from langchain_community.document_loaders import TextLoader             # For changed knowledge documnet into format acceptable format for LangChain process.  Untuk mengubah knowledge document jadi format yang bisa diproses LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter     # For chunking
from langchain_huggingface import HuggingFaceEmbeddings                 # For embedding
from langchain_community.vectorstores import FAISS                      # Vector database
from langchain_groq import ChatGroq                                     # Connect to API Groq
from langchain.chains import RetrievalQA                                # Orkestrator
from langchain.prompts import PromptTemplate                            # For reading and formatting file system_prompt.txt

load_dotenv()

# ── Konfigurasi ────────────────────────────────────────────────────────

# All katalog menu
DATA_PATH = "daftar_menu.txt"

# System prompt location
SYSTEM_PROMPT_PATH = "system_prompt.txt"

# Model embedding: changed text into number vectors
# Use multilingual model for understood Indonesia Languange
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Use LLM Model for answering questions
LLM_MODEL = "llama-3.3-70b-versatile"

# Size excerpts from each text (in characters)
CHUNK_SIZE = 800

# Overlap between continous contextual segments
CHUNK_OVERLAP = 100

# How many points are deducted for each questions
TOP_K_RESULTS = 4


# ── Load System Prompt from File ───────────────────────────────────────

def load_system_prompt(path: str) -> str:
    """
    Reading file system_prompt.txt and give back as string.

    System prompt saved in different file for:
    - Acceptable for modification without taken Python code
    - Safer: system instuctions separated from programming logics 
    - Cleaner: Python code just focused on logics not long text  

    Format file SML-style for:
    - Explained structure ( Kejelasan struktur (Each section has an opening and closing tag)
    - Security: The LLM is trained to treat XML tags as structural boundaries
    - Readability: Anyone who opens the file will immediately understand what each section is about
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT_TEMPLATE = load_system_prompt(SYSTEM_PROMPT_PATH)


# ── Fungsi Build Pipeline ──────────────────────────────────────────────

def build_rag_pipeline():
    """
    Build RAG pipline complete from scratch.

    Mengembalikan:
    - chain: objek RetrievalQA yang siap menerima pertanyaan
    - num_chunks: jumlah potongan teks yang berhasil diindeks
    """

    # ------------------------------------------------------------------
    # First Step: LOAD — Reading the product catalog file
    # ------------------------------------------------------------------
    # Text Loader reads plain text files and converts them into objects
    # Documents that can be processed by LangChain.
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # ------------------------------------------------------------------
    # Second Step: CHUNK — Cutting documents into small pieces
    # ------------------------------------------------------------------
    # Why is chunking necessary??
    # LLM have a limit on the length of text they can process at one time.
    # By filtering, we can select ONLY the relevant parts
    # to send to the LLM—which is more efficient and accurate.
    #
    # separators: Order of priority for separators when cutting
    # "\n---\n" = the dividing line between products in our catalog
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)

    # ------------------------------------------------------------------
    # Third Step: EMBED — Converting text to numerical vectors
    # ------------------------------------------------------------------
    # Embedding is the process of converting text into a sequence of numbers (a vector)
    # that represents the “meaning” of the text.
    #  Texts with similar meanings will produce vectors that are close to each other.
    #
    # Note: The model will be downloaded automatically the first time (~400MB).
    # After that, it is stored in the local cache.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # ------------------------------------------------------------------
    # Fourth Step: STORE — Storing vectors in FAISS
    # ------------------------------------------------------------------
    # FAISS (Facebook AI Similarity Search) is very fast database vektor
    # for finding similarities between texts.
    # All chunks and their vectors are stored here in local memory.
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ------------------------------------------------------------------
    # Fifth Step: RETRIEVER — Setting up a search mechanism
    # ------------------------------------------------------------------
    # A retriever is a component that receives user queries,
    # converts them into vectors, and then searches for
    # the most similar chunks  in the FAISS vector store.
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    # ------------------------------------------------------------------
    # Sixth Step: Secure the API
    # ------------------------------------------------------------------
    # This code taken from Streamlit Secrets
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    else:
        pass #Lock a Groq call

    # ------------------------------------------------------------------
    # Seventh Step: LLM — Initializating a languange model using Groq
    # ------------------------------------------------------------------
    # Groq is platform an access into LLM with
    # ultra hingh speed interfence.
    # Temperature 0.3 = relatively consistent and factual answers
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # ------------------------------------------------------------------
    # Eighth Step: PROMPT — Instruction template for LLM
    # ------------------------------------------------------------------
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # ------------------------------------------------------------------
    # Ninth: CHAIN — Combining all the components
    # ------------------------------------------------------------------
    # RetrievalQA combines a Retriever, an LLM, and a Prompt into
    # a single pipeline that can directly accept questions.
    #
    # All chunks that are extracted are immediately
    # included in a single prompt (suitable for a small number of chunks)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, len(chunks)

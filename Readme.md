# Link Streamlit
https://finalprojectppkd-xmzabwtfrnh73pn7acxysy.streamlit.app/


# Foods and Beverages Advisor

AI-powered chatbot for restaurant sales and marketing teams using Retrieval-Augmented Generation (RAG). The application helps users discover menu recommendations, compare products, answer restaurant policy questions, and provide accurate information directly from the official restaurant catalog.

Built with LangChain, FAISS, Groq LLM, and Streamlit.
## Advisor Sistem

```
Katalog Produk (TXT)
       |
  Document Loader        <- Membaca file katalog
       |
  Text Splitter          <- Memotong dokumen jadi chunk kecil
       |
HuggingFace Embeddings   <- Mengubah teks jadi vektor angka
       |
  FAISS Vector Store     <- Menyimpan vektor untuk pencarian cepat
       |
    Retriever            <- Mencari chunk paling relevan saat ada query
       |
 Groq LLM (Llama 3.3)   <- Merangkai jawaban dari chunk yang diambil
       |
  Jawaban Final
```

## Tech Stack

- LLM: Groq API + Llama 3.3 70B Versatile
- RAG Framework: LangChain
- Embeddings: HuggingFace (paraphrase-multilingual-MiniLM-L12-v2)
- Vector Store: FAISS (lokal, tidak perlu server)
- UI: Streamlit

## Cara Setup dan Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Buat file .env

Salin file `.env.example` dan rename menjadi `.env`:

```bash
cp .env.example .env
```

Lalu isi API key Groq di file `.env`:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

Cara mendapatkan API key:
1. Buka https://console.groq.com
2. Daftar atau login
3. Klik "Create API Key"
4. Salin dan tempelkan ke file .env

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

Buka browser dan akses http://localhost:8501

## Struktur Project

```
smartphone-advisor/
├── app.py                  <- Antarmuka Streamlit (entry point)
├── rag_pipeline.py         <- Logika RAG LangChain
├── requirements.txt        <- Daftar library yang dibutuhkan
├── .env.example            <- Template konfigurasi API key
├── .env                    <- API key (JANGAN di-commit ke GitHub)
├── .gitignore
├── README.md
└── data/
    └── restaurant_catalog.txt  <- Knowledge base produk
```

## Knowledge Base

1. Company Profile
2. Restaurant Vision & Mission
3. Main Menu
4. Vegetarian Menu
5. Drinks
6. Value Packs
7. Promotions
8. Discounts
9. Restaurant Policies
10. Frequently Asked Questions (FAQ)

## Contoh Pertanyaan

- "Rekomendasikan main menu"
- "Rekomendasikan vegetarian menu"
- "Menu apa yang cocok untuk penyuka seafood?"
- "Apa menu yang paling murah di Main Menu?"
- "Menu apa yang paling cocok untuk penyuka mie?"

## Deployment ke Streamlit Community Cloud

1. Upload project ke GitHub (pastikan .env tidak ikut ter-commit)
2. Buka https://share.streamlit.io
3. Hubungkan dengan repository GitHub
4. Tambahkan GROQ_API_KEY di bagian Secrets:
   ```
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
   ```
5. Deploy

Catatan: File .env tidak digunakan di Streamlit Cloud.
API key dibaca dari Secrets yang dikonfigurasi di dashboard.

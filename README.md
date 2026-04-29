
# 📄 AI PDF Summarizer & RAG Q&A

A "smart" PDF tool that lets you **upload any PDF document** and either get an
AI-generated summary or ask natural-language questions about its content.
Built with **Streamlit**, **LangChain**, **FAISS**, and the **OpenAI API** using a
Retrieval-Augmented Generation (RAG) architecture.



https://github.com/user-attachments/assets/ff2741c2-97a5-4a6d-9080-b72d34652ece



---

## ✨ Features

| Feature | Details |
|---|---|
| **PDF text extraction** | Powered by `pypdf` |
| **RAG pipeline** | LangChain splits text into chunks → OpenAI embeds them → FAISS stores and retrieves relevant chunks |
| **AI Q&A** | Ask questions; the model answers using only the relevant parts of your document |
| **Auto-summary** | One-click summary of the entire document |
| **Domain modes** | Switch between *General*, *Medical Research*, and *Legal Contracts* system prompts for more targeted responses |
| **Secure** | API key entered in the sidebar – never stored or logged |

---

## 🛠️ Tech Stack

- **UI**: [Streamlit](https://streamlit.io/)
- **PDF parsing**: [pypdf](https://pypdf.readthedocs.io/)
- **LLM orchestration**: [LangChain](https://python.langchain.com/)
- **Embeddings & chat**: [OpenAI](https://platform.openai.com/) (`text-embedding-3-small` + `gpt-4o-mini`)
- **Vector store**: [FAISS](https://faiss.ai/) (in-memory, rebuilt on each upload)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1 – Clone the repository

```bash
git clone https://github.com/bmehta/ai-pdf-summarizer.git
cd ai-pdf-summarizer
```

### 2 – Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### 4 – Configure your API key

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-...
```

Alternatively, you can paste the key directly into the sidebar at runtime – no
`.env` file required.

### 5 – Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
ai-pdf-summarizer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Python / Streamlit ignores
└── README.md
```

---

## 🔬 How It Works (RAG Pipeline)

```
PDF upload
    │
    ▼
Text extraction (pypdf)
    │
    ▼
Text chunking (LangChain RecursiveCharacterTextSplitter)
    │  chunk_size=1000, overlap=200
    ▼
Embedding (OpenAI text-embedding-3-small)
    │
    ▼
FAISS vector index (in-memory)
    │
    ├─── Summary ──▶ Top-6 chunks → GPT-4o-mini → Summary
    │
    └─── Q&A ─────▶ Top-5 similar chunks → GPT-4o-mini → Answer
```

---

## 🏥⚖️ Domain Modes

| Mode | Best for | System prompt focus |
|---|---|---|
| **General** | Any document | Neutral helpful assistant |
| **Medical Research** | Clinical papers, studies | Medical terminology, abbreviations, study limitations |
| **Legal Contracts** | NDAs, agreements, leases | Clause identification, obligations, risk flags |

---

## 📦 Open-Source Dataset Ideas

| Domain | Source |
|---|---|
| Medical papers | [PubMed Central Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) |
| Legal contracts | [EDGAR SEC filings](https://www.sec.gov/cgi-bin/browse-edgar) |
| General documents | [arXiv](https://arxiv.org/) |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

## 📄 License

MIT

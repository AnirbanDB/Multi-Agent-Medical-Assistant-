<div align="center">

# ⚕️ Multi-Agent Medical Assistant

**An AI-powered multi-agent system for medical diagnosis, knowledge retrieval, and patient interaction.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-teal?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal?style=for-the-badge&logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-1.18+-red?style=for-the-badge&logo=qdrant)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)

</div>

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [How It Works — Request Lifecycle](#-how-it-works--request-lifecycle)
- [The Agents](#-the-agents)
- [RAG Pipeline](#-rag-pipeline-deep-dive)
- [Models & Datasets](#-models--datasets)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Roadmap](#-roadmap)
- [Credits & Attribution](#-credits--attribution)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 📌 Overview

The **Multi-Agent Medical Assistant** is an AI chatbot that routes each user request to a specialized agent best suited to handle it. A single orchestration graph decides whether a query should be answered by conversation, retrieved from an internal medical knowledge base (RAG), searched on the live web, or analyzed by a medical computer-vision model — and applies safety guardrails and human-in-the-loop validation along the way.

**Key capabilities:**
- 🤖 **Multi-agent orchestration** with confidence-based routing and agent-to-agent handoff (LangGraph)
- 📚 **Hybrid RAG** over medical documents (dense + sparse retrieval, cross-encoder reranking)
- 🌐 **Live web search** fallback to reduce hallucinations
- 🖼️ **Medical imaging** — COVID chest X-ray classification & skin-lesion segmentation
- 🎙️ **Voice interaction** — speech-to-text and text-to-speech
- 🛡️ **Input/output guardrails** and **human-in-the-loop** validation for diagnoses

> This build runs on a **fully free / open-source model stack** (Groq-hosted Llama, local HuggingFace embeddings, Qdrant Cloud, ElevenLabs & Tavily free tiers) — no paid API required. Every model is configurable in `config.py`.

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  BROWSER UI  (HTML / CSS / JS)                                       │
│  text · image upload · 🎙️ mic · 🔊 speaker                           │
└───────────────────────────────┬────────────────────────────────────┘
                                 │  REST (HTTP)
┌───────────────────────────────▼────────────────────────────────────┐
│  FastAPI SERVER  (app.py)  — single entry point                     │
│  /chat   /upload   /transcribe   /generate-speech   /validate        │
└───────────────────────────────┬────────────────────────────────────┘
                                 │  process_query()
┌───────────────────────────────▼────────────────────────────────────┐
│  ORCHESTRATOR  —  LangGraph state machine (agents/agent_decision.py) │
│                                                                      │
│   input guardrail → image-type check → route → run agent →           │
│                         → human validation → output guardrail        │
└───────┬───────────┬────────────┬──────────────┬───────────────┬─────┘
        ▼           ▼            ▼              ▼               ▼
  Conversation    RAG        Web Search    Chest X-ray     Skin Lesion
   (Llama LLM)  (Llama +    (Llama +        (DenseNet121    (U-Net
                bge + Qdrant  Tavily)        classifier)     segmentation)
                + reranker)
```

**Three knowledge sources feed the agents:**
| Source | Purpose |
|--------|---------|
| **Groq-hosted LLM** | all natural-language reasoning, routing, guardrails |
| **Qdrant Cloud + local embedder/reranker** | internal medical knowledge (RAG memory) |
| **Tavily** | real-time web information |

Only lightweight models run **locally** (embeddings, reranker, two CV models); all LLM calls and vector storage are **cloud-hosted**, so the system runs comfortably on a laptop.

---

## 🔄 How It Works — Request Lifecycle

Every text or image query flows through the **same LangGraph pipeline**:

1. **Input guardrail** — the LLM screens the text for unsafe/off-topic/PII/injection content; blocks early if needed.
2. **Image-type check** *(if an image is uploaded)* — a vision-capable LLM classifies it as *chest X-ray / skin lesion / brain MRI / non-medical*.
3. **Routing** — the LLM triage node reads the query + recent context + image type and returns a JSON decision (`agent`, `reasoning`, `confidence`). Low confidence (< 0.85) safely defaults to RAG.
4. **Agent execution** — the selected specialist agent runs (see below).
5. **Human-in-the-loop validation** — for medical-imaging diagnoses, the system pauses for a human **Yes/No** confirmation before finalizing.
6. **Output guardrail** — the LLM reviews the final response for safety before it reaches the user.

Conversation history is preserved across turns via LangGraph's `MemorySaver`.

---

## 🧠 The Agents

| Agent | What it does | Engine |
|-------|--------------|--------|
| **Conversation** | General chat & medical Q&A | Groq LLM |
| **RAG** | Answers from ingested medical literature; **auto-hands off to Web Search** when retrieval confidence < 0.40 | Groq LLM + Qdrant + bge embeddings + reranker |
| **Web Search** | Reformulates query → Tavily search → LLM-summarized answer | Groq LLM + Tavily |
| **Chest X-ray** | Classifies an uploaded chest X-ray as **COVID-19 vs Normal** | DenseNet121 (PyTorch) |
| **Skin Lesion** | Produces a **segmentation mask** of the lesion, overlaid on the image | U-Net (PyTorch) |

> **Brain Tumor agent** is scaffolded in the routing menu but **not yet implemented** (see [Roadmap](#-roadmap)).

---

## 📖 RAG Pipeline (Deep Dive)

**Ingestion** (`ingest_rag_data.py` → `agents/rag_agent/`):
1. **Parse** the PDF with **Docling** (extracts text, tables, page images, figures).
2. **Summarize figures** into text using a vision LLM.
3. **Semantic chunking** — an LLM decides split points so chunks respect topic boundaries (256–512 words).
4. **Embed** each chunk with **BAAI/bge-large-en-v1.5** (1024-dim) and store in **Qdrant** as both a dense vector and a **BM25 sparse** vector; full chunk text is kept in a local doc store.

**Querying:**
1. **Query expansion** — LLM adds relevant medical terminology.
2. **Hybrid retrieval** — Qdrant combines dense (semantic) + sparse (BM25 keyword) search → top 5 chunks.
3. **Reranking** — a **cross-encoder (ms-marco-TinyBERT-L-6)** re-scores and keeps the top 3.
4. **Answer generation** — the LLM answers strictly from retrieved context, appending source links and reference figures.
5. **Confidence** — average retrieval score decides whether to fall back to Web Search (anti-hallucination).

---

## 🧬 Models & Datasets

### Models
| Task | Model | Runs on | Notes |
|------|-------|---------|-------|
| LLM (reasoning/routing/guardrails) | **Llama 3.3 70B** via Groq | Cloud | configurable |
| Vision (image type + figure summaries) | **Llama-4 Scout** via Groq | Cloud | multimodal |
| Text embeddings | **BAAI/bge-large-en-v1.5** | Local | 1024-dim |
| Reranker | **cross-encoder/ms-marco-TinyBERT-L-6** | Local | — |
| Chest X-ray classification | **DenseNet121** (transfer-learned) | Local | COVID-19 vs Normal |
| Skin lesion segmentation | **U-Net** | Local | binary mask, 256×256 |
| Speech-to-text | **ElevenLabs Scribe** (`scribe_v1`) | Cloud | — |
| Text-to-speech | **ElevenLabs Flash** (`eleven_flash_v2_5`) | Cloud | low-cost, low-latency |

> ⚠️ The pre-trained CV weights are external artifacts (chest X-ray weights ship in the repo; the skin-lesion U-Net checkpoint downloads on first run). This repository contains **inference code only** — no training scripts.

### Datasets used to train the imaging models
The imaging models are trained/compatible with the following public datasets:

- **Chest X-ray — COVID-19 positive images:** [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Chest X-ray — Normal images:** [Chest X-Ray Images (Pneumonia) — Kermany/Mooney (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Chest X-ray — pre-combined COVID/Normal option:** [Chest X-ray (COVID-19 & Pneumonia) (Kaggle)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
- **Skin Lesion segmentation:** [ISIC 2018 Task 1 — Lesion Boundary Segmentation (Kaggle mirror)](https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation) · [Official ISIC 2018 challenge](https://challenge2018.isic-archive.com/task1/)

Sample test images for each model are provided under [`sample_images/`](sample_images/).

### RAG knowledge base
The vector DB is built from medical literature PDFs in [`data/raw/`](data/raw/) (brain tumors, COVID chest X-ray, skin lesion). Add your own PDFs and re-ingest to extend the assistant's knowledge.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| Orchestration | LangGraph + LangChain |
| LLM | Groq (Llama 3.3 70B / Llama-4 Scout) — *swappable* |
| Embeddings | HuggingFace `bge-large-en-v1.5` (local) |
| Vector DB | Qdrant (Cloud or local) — hybrid dense + BM25 |
| Reranker | HuggingFace cross-encoder |
| Document parsing | Docling |
| Medical imaging | PyTorch (DenseNet121, U-Net) |
| Web search | Tavily |
| Speech | ElevenLabs (STT + TTS) |
| Frontend | HTML / CSS / JavaScript |
| Deployment | Docker + GitHub Actions |

---

## 📁 Project Structure

```
Multi-Agent-Medical-Assistant/
├── app.py                     # FastAPI server (entry point)
├── config.py                  # All model & system configuration
├── ingest_rag_data.py         # Ingest PDFs into the vector DB
├── agents/
│   ├── agent_decision.py      # LangGraph orchestrator
│   ├── rag_agent/             # Parse → chunk → embed → retrieve → rerank → answer
│   ├── web_search_processor_agent/   # Tavily web search
│   ├── image_analysis_agent/  # Chest X-ray, skin lesion, (brain tumor TBD)
│   └── guardrails/            # Input/output safety filters
├── data/                      # raw PDFs, parsed docs, vector store
├── sample_images/             # Example images to try the CV agents
├── templates/index.html       # Chat UI
└── requirements.txt
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- `ffmpeg` (required for speech features)

### 1. Clone
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/Multi-Agent-Medical-Assistant.git
cd Multi-Agent-Medical-Assistant
```

### 2. Environment
```bash
conda create -n medassist python=3.11 && conda activate medassist
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

### 3. Configure API keys
Create a `.env` file in the project root (this file is git-ignored — never commit it):

```bash
# LLM — Groq (free tier)
GROQ_API_KEY=your_groq_key
GROQ_MODEL_NAME=llama-3.3-70b-versatile
GROQ_VISION_MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct

# Embeddings — local HuggingFace model (no key needed at inference)
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5

# Speech (ElevenLabs free tier)
ELEVEN_LABS_API_KEY=your_elevenlabs_key

# Web search (Tavily free tier)
TAVILY_API_KEY=your_tavily_key

# HuggingFace token (to download embedding + reranker models)
HUGGINGFACE_TOKEN=your_hf_token

# Qdrant — leave blank to use the bundled local DB, or set for Qdrant Cloud
QDRANT_URL=
QDRANT_API_KEY=
```

> Using a different provider (OpenAI, Gemini, Ollama, etc.)? Swap the model definitions in `config.py`. If you change the embedding model, update `embedding_dim` in `config.py` and **re-ingest** the vector DB.

### 4. Run
```bash
python app.py
```
Open **http://localhost:8000**.

### 5. Ingest documents into the vector DB
```bash
# single file
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
# entire directory
python ingest_rag_data.py --dir ./data/raw
```

### 🐳 Docker
```bash
docker build -t medical-assistant .
docker run -d --name medical-assistant-app -p 8000:8000 --env-file .env \
  -v $(pwd)/data:/app/data -v $(pwd)/uploads:/app/uploads medical-assistant
```

> **First run note:** several models download on first launch (embeddings, reranker, Docling parsing models, skin-lesion checkpoint). Be patient and watch the console. Ensure you have a few GB of free disk.

---

## 🧪 Usage

| Try | How |
|-----|-----|
| 💬 Conversation | Ask a general question |
| 📚 RAG | Ask about ingested topics (e.g. brain tumors) |
| 🌐 Web search | Ask about recent medical developments |
| 🫁 Chest X-ray | Upload an image from `sample_images/chest_x-ray_covid_and_normal/` |
| 🔬 Skin lesion | Upload an image from `sample_images/skin_lesion_images/` |
| 🎙️ Voice | Use the mic / speaker controls |

---

## 🗺️ Roadmap

- [ ] Brain tumor detection/segmentation model integration (currently scaffolded, not implemented)
- [ ] Quantitative evaluation suite (accuracy, Dice/IoU) for the imaging models
- [ ] Optional local LLM (Ollama) and local STT/TTS (Whisper / Piper) backends

---

## 🙏 Credits & Attribution

This project is **based on and adapted from** the original open-source work by **Souvik Majumder** — [github.com/souvikmajumder26/Multi-Agent-Medical-Assistant](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant), licensed under Apache-2.0.

**Modifications in this fork include:** migration to a fully free / open-source model stack (Groq-hosted Llama for LLM & vision, local `bge-large` embeddings, updated Qdrant Cloud integration), cost-optimized ElevenLabs voice models, and expanded documentation.

Please retain this attribution and the `LICENSE` file when redistributing, as required by Apache-2.0.

---

## ⚖️ License

Licensed under the **Apache-2.0 License**. See [LICENSE](LICENSE).

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It is **not** a medical device and must **not** be used for real clinical diagnosis or treatment decisions. Always consult a licensed healthcare professional.

# Multimodal Laptop Two Stage RAG Assistant

A local multimodal Retrieval-Augmented Generation (RAG) system that uses CLIP for embedding laptop images and text specifications, stores them in ChromaDB, and retrieves the most relevant specs through a Streamlit UI.

---

## Status & Technologies

![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-2ca5a5?logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b?logo=streamlit&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Storage-blue)
![CLIP](https://img.shields.io/badge/CLIP-ViT%20Base-green)
![CrossEncoder](https://img.shields.io/badge/Cross--Encoder-MiniLM--L--6--v2-orange)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikit-learn&logoColor=white)
![Torch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)

---

## Features

- Multimodal input: image, text, or both
- Embeds laptops specs using CLIP (text + image)
- Stores embeddings in local ChromaDB
- Uses a cross-encoder for intelligent reranking
- Query via web UI (Streamlit)

---

## Folder Structure

```
.
multimodal_rag_demo/
│
├── run_once.py                         # Builds the ChromaDB index from text + image
├── requirements.txt                    # Updated dependencies (no Ollama, using HF CLIP)
│
├── rag/                                # Backend logic
│   ├── __init__.py
│   ├── config.py                       # Folder paths and constants
│   ├── embedding_utils.py              # CLIP embedding (text + image) using Hugging Face
│   ├── index_builder.py                # Creates + populates chromadb collections
│   ├── loaders.py                      # Loads .txt and image paths from folders
│   └── reranker.py                     # Uses cross-encoder to rerank top results
│
├── frontend/                           # UI logic
│   └── app.py                          # Streamlit app with 3 modes: image, text, combined
│
├── data/
│   ├── documents/                      # Text specs (e.g., Dell.txt, Lenovo.txt)
│   │   └── ...
│   └── images/                         # Laptop images (e.g., Dell.jpg, Lenovo.jpg)
│       └── ...
│
└── chromadb_storage/                   # Persistent vector store created by Chroma
    ├── chroma-collections.parquet
    ├── chroma-index.parquet
    └── index/

```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/laptop-rag-assistant.git
cd laptop-rag-assistant
```

2. Create a virtual environment and activate it

```bash
python -m venv env
source env/bin/activate   # On Windows: .\env\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Your Data

- Place `.txt` specification files into `data/documents/`
- Place corresponding `.jpg` or `.png` laptop images into `data/images/`

### 2. Generate Embeddings and Build Index

```bash
python run_once.py
```

### 3. Launch the UI

```bash
streamlit run app.py
```

---

## Available Modes

- **Image → Specs**: Upload an image and get matching specs
- **Image + Text → Specs**: Improve match with description
- **Text → Image + Specs**: Describe and retrieve closest match

---

## Models Used

- `openai/clip-vit-base-patch32` – for multimodal embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` – for semantic reranking

---

## Dependencies

Contents of `requirements.txt`:

```
streamlit
chromadb
llama-index
langchain
langchain-community
langchain-nomic
sentence-transformers
transformers
pillow
pypdf
scikit-learn
numpy
```

---

## License



---

## Acknowledgements

- Hugging Face for model access
- LangChain/ChromaDB for vector search
- Streamlit for interactive UI


# 🧠 NIC Custom LLM with Multi-Model Support + RAG

A powerful multi-model Language Model system built for **National Industries Company (NIC)**. It combines multiple quantized models (GGUF), **Retrieval-Augmented Generation (RAG)**, and advanced **prompt engineering** to deliver smart, accurate, and context-aware responses from internal company data.

---

## 📌 Highlights

- 🔄 **Multi-model support**: Easily switch between Phi-2, Mistral, DeepSeek, LLaMA, and more
- 🧠 **Token-aware chunking** using Hugging Face's tokenizer
- 📚 **RAG (Retrieval-Augmented Generation)** to ground LLM responses with real company data
- 💬 **Prompt engineering** for better relevance and tone
- 🗃️ **ChromaDB** for persistent, fast document vector search
- 🌐 Local setup — No cloud dependencies

---

## 🧩 Models Used

Models are stored in the `models/` directory in GGUF format:

```
models/
├── deepseek-llm-7b-chat.Q4_K_M.gguf
├── DeepSeek-V2-Lite-Chat-Q4_K_M.gguf
├── llama-2-7b-chat.Q4_K_M.gguf
├── Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
├── phi-2-q4_k_m.gguf
```

Use `ctransformers`, `llama-cpp-python`, or `exllama` to load locally.

---

## 📁 Project Files

| File            | Description                                                      |
|------------------|------------------------------------------------------------------|
| `chatbot.py`     | Main chatbot orchestrator                                        |
| `deepseek.py`    | DeepSeek model wrapper                                           |
| `phi.py`         | Phi-2 model wrapper                                              |
| `model.py`       | Unified loader for GGUF models                                   |
| `nicQA.py`       | Handles RAG + prompting logic                                    |
| `nicVDB.py`      | Vector database logic using Chroma                               |
| `prepare_data.py`| Loads + chunks NIC documents and embeds them                    |
| `data/`          | Contains `nic_data.txt` with company info                        |
| `models/`        | Quantized models in GGUF format                                  |

---

## 📚 Data Preparation & Embedding

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
loader = TextLoader("data/nic_data.txt", encoding="utf-8")
documents = loader.load()

def split_by_tokens(text, tokenizer, max_tokens=400, overlap_tokens=50):
    tokens = tokenizer.tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
        i += max_tokens - overlap_tokens
    return chunks

token_chunks = [chunk for doc in documents for chunk in split_by_tokens(doc.page_content, tokenizer)]
docs = [Document(page_content=chunk) for chunk in token_chunks]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
Chroma.from_documents(docs, embeddings, persist_directory="./vector_db")
```

---

## 🧠 Context Retrieval with NICVectorDB

NICVectorDB wraps **ChromaDB** for vector-based document search using multilingual sentence embeddings.

#### 📦 `nicVDB.py`

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class NICVectorDB:
    def __init__(self, db_path="./vector_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

    def retrieve_context(self, query, k=5):
        results = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])
```

Use it to enhance model responses by injecting real NIC context.

---

## ⚙️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**
   ```bash
   python prepare_data.py
   ```

3. **Start chatbot**
   ```bash
   python chatbot.py
   ```

4. **Choose your model** in `model.py` (e.g. `phi`, `deepseek`, `llama`)

---

## 📌 Example Prompt

```text
You are an assistant at NIC. Use the context below to answer precisely.
---
{retrieved_chunks}
---
Q: {user_question}
A:
```

---

## 📈 Future Upgrades

- Web UI with Flask or Gradio
- Arabic language support
- Whisper voice input
- Response logging and analytics

---

## 👤 Author

**Ahmed Zendah**  
AI Engineer | NIC Projects Dept  
🔗 GitHub: [zendah21](https://github.com/zendah21)

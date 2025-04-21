from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Token-based splitting function
def split_by_tokens(text, tokenizer, max_tokens=400, overlap_tokens=50):
    tokens = tokenizer.tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
        i += max_tokens - overlap_tokens
    return chunks

# Load data
file_path = "data/nic_data.txt"
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# Split data based on tokens
token_chunks = []
for doc in documents:
    token_chunks.extend(split_by_tokens(doc.page_content, tokenizer))

# Prepare documents for embedding
from langchain.docstore.document import Document
docs = [Document(page_content=chunk) for chunk in token_chunks]

# Embedding and vector store setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./vector_db")

print(f"✅ Stored {len(docs)} token-based chunks in ChromaDB!")

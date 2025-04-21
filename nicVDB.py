from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings

class NICVectorDB:
    def __init__(self, db_path="./vector_db"):
        """
        Initialize ChromaDB with stored embeddings.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

    def retrieve_context(self, query, k=5):
        """
        Retrieve the most relevant text chunks from NIC data.
        :param query: User's search query.
        :param k: Number of top results to retrieve.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        retrieved_text = "\n\n".join([doc.page_content for doc in results])

        #print(f"\n🔍 Retrieved Context for '{query}':\n{retrieved_text}\n")  # Debugging
        return retrieved_text

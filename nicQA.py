from deepseek import DeepSeek
from nicVDB import NICVectorDB
import re
import time
import logging
from transformers import AutoTokenizer

logging.basicConfig(
    filename='nicqa_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def truncate_context(text, max_tokens=400):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)


def clean_answer(answer):
    # More robust clean-up
    clean_text = re.split(r"(###|\"\"\")", answer)[0].strip()
    return clean_text

class NICQA:
    def __init__(self, model_path="models/deepseek-llm-7b-chat.Q4_K_M.gguf", db_path="./vector_db"):
        self.llm = DeepSeek(model_path)
        self.vector_db = NICVectorDB(db_path)

    def format_history(self, history, max_turns=3):
        """
        Formats conversation history for better model understanding.
        """
        history = history[-max_turns:]
        formatted_history = ""
        for h in history:
            formatted_history += f"User: {h['user_question']}\nAI: {h['chatBot_answer']}\n"
        return formatted_history

    def ask(self, question, temperature=0.7, max_tokens=300, top_p=0.9, history=None):
        if history is None:
            history = []

        logging.info(f"\n=== New Query ===\nQuestion: {question}")

        retrieval_start = time.time()
        nic_context = self.vector_db.retrieve_context(question)
        retrieval_end = time.time()

        nic_context = truncate_context(nic_context, max_tokens=500)

        formatted_history = self.format_history(history)

        prompt = f"""
        You are an AI assistant for National Industries Company (NIC).

        ### Guidelines:
        - Respond in the same language as the user's question (Arabic or English).
        - Use **Company Information** provided to answer precisely.
        - If unsure, clearly state information isn't available.
        - Respond professionally and politely.
        - Handle greetings and farewells appropriately.
        - If the user is disrespectful, respond in the user's language:
          - Arabic: "سيتم الإبلاغ عن أي سلوك غير محترم."
          - English: "Disrespectful behavior will be reported."

        ### Company Information:
        {nic_context}

        ### Conversation History:
        {formatted_history}

        ### Current Question:
        {question}

        ### Answer:
        """

        generation_start = time.time()
        raw_answer = self.llm.generate_response(prompt, temperature, max_tokens, top_p)
        generation_end = time.time()

        cleaned_answer = clean_answer(raw_answer)

        retrieval_time = retrieval_end - retrieval_start
        generation_time = generation_end - generation_start
        total_time = retrieval_time + generation_time

        logging.info(f"Retrieval Time: {retrieval_time:.2f}s")
        logging.info(f"LLM Generation Time: {generation_time:.2f}s")
        logging.info(f"Total Time: {total_time:.2f}s")

        print(f"\n⏱ Retrieval: {retrieval_time:.2f}s | 🧠 Generation: {generation_time:.2f}s | ⚡ Total: {total_time:.2f}s")

        return cleaned_answer

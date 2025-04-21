import os
import psutil
import panel as pn
import param
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch

# ✅ Dynamically Adjust LLM Hyperparameters Based on Available System Resources
available_ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
n_ctx = 4096 if available_ram >= 32 else 2048 if available_ram >= 16 else 1024
num_threads = os.cpu_count() // 2

# ✅ Load FAISS Vector Store (Semantic Search)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)

# ✅ Load Local GGUF Llama Model (Optimized)
llm = LlamaCpp(
    model_path="models/phi-2-q4_k_m.gguf",
    temperature=0.7,
    max_tokens=100,  # ✅ Reduced token limit for faster response
    n_batch=4,  # ✅ Lower batch size improves speed
    n_ctx=n_ctx,  # ✅ Dynamic context window
    n_threads=num_threads,  # ✅ Multi-threading for performance
    top_p=0.8,  # ✅ Reduces randomness
    verbose=False
)

# ✅ Set Up Conversational Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Function to Load PDFs into a Vector Database
def load_db(file):
    """Loads a PDF document, splits text, and stores it in a vector database."""
    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Create in-memory vector database
    db = DocArrayInMemorySearch.from_documents(docs, embeddings_model)

    # Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return qa_chain

# ✅ Chatbot Class with Panel GUI
class Chatbot(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(Chatbot, self).__init__(**params)
        self.panels = []
        self.loaded_file = "docs/sample.pdf"  # Default file
        self.qa = load_db(self.loaded_file)

    def call_load_db(self, count):
        """Handles loading new files into the database."""
        if count == 0 or file_input.value is None:
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # Save file locally
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf")
            button_load.button_style = "solid"
        self.clear_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def conv_chain(self, query):
        """Handles user queries and retrieves relevant information."""
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)

        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']

        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  # Clear input
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends('db_query')
    def get_last_query(self):
        """Displays the last query sent to the vector database."""
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown("Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("No DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown("DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query)
        )

    @param.depends('db_response')
    def get_sources(self):
        """Displays the retrieved source documents."""
        if not self.db_response:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No Sources Found")), width=600, scroll=True)
        response_list = [pn.Row(pn.pane.Markdown("Retrieved Documents:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            response_list.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*response_list, width=600, scroll=True)

    @param.depends('conv_chain', 'clear_history')
    def get_chat_history(self):
        """Displays the chat history."""
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        history_list = [pn.Row(pn.pane.Markdown("Current Chat History:", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            history_list.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*history_list, width=600, scroll=True)

    def clear_history(self, count=0):
        """Clears the chat history."""
        self.chat_history = []
        return

# ✅ Initialize Chatbot
chatbot = Chatbot()

# ✅ Panel UI Components
file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(chatbot.clear_history)

inp = pn.widgets.TextInput(placeholder='Enter text here…')

bound_button_load = pn.bind(chatbot.call_load_db, button_load.param.clicks)
conversation = pn.bind(chatbot.conv_chain, inp)

# ✅ GUI Layout
tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2 = pn.Column(
    pn.panel(chatbot.get_last_query),
    pn.layout.Divider(),
    pn.panel(chatbot.get_sources),
)
tab3 = pn.Column(
    pn.panel(chatbot.get_chat_history),
    pn.layout.Divider(),
)
tab4 = pn.Column(
    pn.Row(file_input, button_load, bound_button_load),
    pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history.")),
    pn.layout.Divider(),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3), ('Configure', tab4))
)

dashboard.show()  # ✅ Run the chatbot UI

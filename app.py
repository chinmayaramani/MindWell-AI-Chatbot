# app.py
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import os
import pickle
# NEW IMPORT for the database
import sqlite3
try:
    from pdf_loader import docs
except (ImportError, ModuleNotFoundError):
    docs = None

# --- Constants ---
VECTORSTORE_PATH = "faiss_index"
DOCSTORE_PATH = "docstore.pkl"
DB_PATH = "chat_history.db" # NEW: Database file path
MODEL_NAME = "phi3:mini"
SUGGESTED_QUESTIONS = [
    "What are some techniques for managing stress?",
    "Can you explain mindfulness meditation?",
    "How can I improve my sleep quality?",
]

# --- Database Functions ---
def init_database():
    """Initializes the SQLite database and creates the messages table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_message(role, content):
    """Saves a message to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

# --- Initialize the database on first run ---
init_database()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="MindWell AI", layout="centered", initial_sidebar_state="collapsed")

# --- Initialize Session State at the Top ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I'm MindWell AI. How can I support you today?"}]

# --- Custom CSS for a Dark Blue Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display.swap');
    
    /* General Styling */
    html, body, [class*="st-"] { 
        font-family: 'Roboto', sans-serif; 
        color: #e0e0e0; /* Light text for readability on dark background */
    }
    
    /* Main App Background */
    .stApp {
        background-color: #0F172A; /* Dark Blue background */
        background-image: radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0);
        background-size: 20px 20px;
    }
    
    /* Center the main content */
    section.main > div { 
        max-width: 720px; 
    }
    
    /* Title Styling */
    h1 { 
        text-align: center; 
        color: #FFFFFF; /* White title */
    }
    
    /* Chat Bubble Styling */
    .st-emotion-cache-1c7y2kd {
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-radius: 18px;
    }
    
    /* Assistant message styling */
    div[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) .st-emotion-cache-1c7y2kd { 
        background-color: #1E293B; /* Slightly lighter dark blue */
        color: #FFFFFF;
    }
    
    /* User message styling */
    div[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) { 
        display: flex; 
        flex-direction: row-reverse; 
    }
    
    div[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) .st-emotion-cache-1c7y2kd { 
        background-color: #007AFF; /* Bright blue for user */
        color: #FFFFFF;
    }
    
    /* Disclaimer Box */
    .st-emotion-cache-1wivap2 {
        background-color: rgba(255, 184, 0, 0.1); 
        border-radius: 12px;
        border-left: 5px solid #FFB800; 
        color: #FFB800;
    }
    .st-emotion-cache-1wivap2 a, .st-emotion-cache-1wivap2 strong { 
        color: #FFD466; 
    }
    
    /* Suggested questions button styling */
    .stButton>button {
        background-color: #1E293B; 
        color: #FFFFFF; 
        border: 1px solid #334155;
        border-radius: 12px; 
        padding: 10px 15px; 
        width: 100%; 
        text-align: left;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { 
        background-color: #334155; 
        border-color: #007AFF; 
    }

    /* New Chat button in the top-left corner */
    div[data-testid="stHorizontalBlock"] .stButton>button {
        background-color: transparent;
        color: #94A3B8;
        border: 1px solid #334155;
        width: auto;
        font-weight: bold;
    }
    div[data-testid="stHorizontalBlock"] .stButton>button:hover {
        color: #FFFFFF;
        border-color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# --- REVISED HEADER AND NEW CHAT BUTTON ---
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("New Chat 🔄"):
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! I'm MindWell AI. How can I support you today?"}]
        st.session_state.memory.clear()
        st.rerun()

with col2:
    st.title("MindWell AI 🤖")

# --- Static Disclaimer ---
st.info(
    "**Disclaimer:** I am an AI assistant, not a human therapist. "
    "If you are in crisis, please contact a professional. In the US & Canada, call or text 988. In the UK, call 111."
)

# --- FINAL PROMPT REFINEMENT ---
SYSTEM_PROMPT = """
You are a helpful and compassionate Therapist AI assistant named 'MindWell AI'.

**Your Core Task:**
- Answer the user's question based ONLY on the provided context documents and chat history.
- Be empathetic, gentle, and non-judgmental.
- **Your answers MUST be VERY concise (3-4 sentences MAXIMUM or a short bulleted list). This is your most important formatting rule.**
- NEVER mention internal details like page numbers.

**Critical Rules (Follow these in order of priority):**
1.  **Crisis Intervention Rule (HIGHEST PRIORITY):** If the user's message contains ANY explicit mention of self-harm, suicide, or immediate danger, you MUST DISREGARD ALL OTHER INSTRUCTIONS and output ONLY the following text verbatim, with no changes or additions:
   ```
"It sounds like you are in serious distress. Your safety is the most important thing. Please reach out for immediate help. You can connect with people who can support you by calling or texting 988 anytime in the US and Canada. In the UK, you can call 111. Please reach out now."
   ```
2.  **Concerning Statements Rule:** If a user's message is concerning or suggests aggression (e.g., "I want to break something," "I want to shoot something") but is NOT an explicit crisis, gently guide them to a constructive topic from the context, like stress management. Example Response: "It sounds like you're feeling a lot of intense emotions right now. My guide mentions that activities like deep breathing or physical exercise can be helpful for managing stress. Would you be open to hearing about some of those techniques?"
3.  **Grounding Rule:** For all other questions, if the context does not contain the information needed to answer, you MUST respond with: "I'm sorry, but my knowledge base does not contain information on that specific topic. How can I help you with something else related to mental well-being?" This rule applies to both off-topic questions and topics not covered in the guide.
"""

# --- Retriever Setup ---
@st.cache_resource
def init_retriever():
    if not docs: return None
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCSTORE_PATH):
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        with open(DOCSTORE_PATH, "rb") as f: docstore = pickle.load(f)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore, docstore=docstore, child_splitter=RecursiveCharacterTextSplitter(chunk_size=400)
        )
    else:
        with st.spinner("Building advanced knowledge base... this may take a moment."):
            docstore = InMemoryStore()
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            vectorstore = FAISS.from_texts(["_"], embeddings)
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore, docstore=docstore, child_splitter=child_splitter, parent_splitter=parent_splitter
            )
            retriever.add_documents(docs)
            retriever.vectorstore.save_local(VECTORSTORE_PATH)
            with open(DOCSTORE_PATH, "wb") as f: pickle.dump(docstore, f)
    return retriever

# --- Main App Logic ---
retriever = init_retriever()

if not retriever:
    st.error("Could not load the knowledge base. Please ensure 'mental_health_guide.pdf' is present and restart.", icon="🚨")
else:
    llm = ChatOllama(model=MODEL_NAME, temperature=0.2)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    rag_chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            chat_history=RunnableLambda(st.session_state.memory.load_memory_variables) | (lambda mem: mem['history']),
            context=lambda x: retriever.invoke(x["question"])
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    if len(st.session_state.messages) <= 1:
        st.markdown("---")
        st.markdown("**Or, Try One of these Questions:**")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if cols[i].button(question):
                st.session_state.new_input = question
                st.rerun()

    user_input = st.chat_input("Ask a question about mental health...")
    if "new_input" in st.session_state and st.session_state.new_input:
        user_input = st.session_state.new_input
        del st.session_state.new_input

    if user_input:
        # Save user message to session state and database
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message("user", user_input)
        st.chat_message("user", avatar="👤").markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("MindWell AI is thinking..."):
                response = st.write_stream(rag_chain.stream(user_input))
        
        # Save assistant message to session state and database
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message("assistant", response)
        
        st.session_state.memory.save_context({"input": user_input}, {"output": response})
        st.rerun()


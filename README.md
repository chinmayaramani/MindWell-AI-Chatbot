MindWell AI Chatbot 🤖
A sophisticated, locally-run AI chatbot designed to provide supportive and context-aware answers to mental health questions. This application leverages a Retrieval-Augmented Generation (RAG) pipeline to ensure all responses are grounded in a curated PDF knowledge base, preventing AI hallucination and providing safe, relevant information.

🌟 Key Features
Advanced RAG Pipeline: Utilizes a ParentDocumentRetriever to ensure the AI has complete context, leading to highly accurate and relevant answers.

100% Local & Private: Runs entirely on your local machine using Ollama, ensuring user privacy and offline functionality.

GPU Accelerated: Automatically leverages NVIDIA GPUs for significantly faster response times.

Persistent Conversation History: Integrates a SQLite database to save all conversations, demonstrating full-stack capabilities and enabling future features.

Robust Safety Protocols: A hardened SYSTEM_PROMPT provides a multi-layered defense against off-topic questions, handles concerning statements with empathy, and includes a critical, verbatim crisis intervention protocol.

Polished User Interface: A custom-styled Streamlit front-end with a professional dark mode, custom chat bubbles, and a clean, intuitive layout.

Fast & Efficient Knowledge Base: Uses a FAISS vector store with persistence for near-instantaneous startups after the initial setup.

🛠️ Tech Stack & Architecture
This project combines a modern AI stack with classic web and database technologies.

Front-End: Streamlit

AI Orchestration: LangChain

LLM Serving: Ollama (running phi3:mini)

Vector Store: FAISS (Facebook AI Similarity Search)

Database: SQLite

PDF Loading: PyPDFLoader

The application follows a standard RAG workflow:

A comprehensive mental health guide (mental_health_guide.pdf) is loaded and split into parent/child chunks.

The child chunks are embedded and stored in a FAISS vector index for fast semantic search.

When a user asks a question, the most relevant child chunks are found, and their corresponding parent chunks are retrieved to provide full context.

The user's question, chat history, and the retrieved context are passed to the LLM via a carefully crafted system prompt.

The AI generates a response, which is streamed back to the user interface and saved to the SQLite database.

🚀 Local Setup and Installation
Follow these steps to run the application on your local machine.

Prerequisites
Python 3.9+

Git

Ollama

1. Clone the Repository
Open your terminal or Git Bash and clone this repository:

git clone [https://github.com/chinmayaramani/MindWell-AI-Chatbot.git](https://github.com/chinmayaramani/MindWell-AI-Chatbot.git)
cd MindWell-AI-Chatbot

2. Set Up the Python Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

3. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

4. Download the AI Model
Pull the phi3:mini model using Ollama. This will download the model to your machine.

ollama pull phi3:mini

5. Add the Knowledge Base
Place your knowledge base PDF in the root of the project folder and ensure it is named mental_health_guide.pdf.

6. (Recommended for Performance) NVIDIA GPU Setup
For the best performance, ensure you have an NVIDIA GPU and have installed the latest NVIDIA Studio Driver. Ollama will automatically use the GPU if it's available.

7. Run the Application
Launch the Streamlit app. The first run will take a moment to build the knowledge base. Subsequent runs will be much faster.

streamlit run app.py

Your chatbot should now be running and accessible in your web browser!

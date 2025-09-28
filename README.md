Mindful AI Therapist 🤖💬
A compassionate and safe AI assistant designed to answer mental health questions using a curated knowledge base. This project showcases a complete Retrieval-Augmented Generation (RAG) pipeline, from local model inference to a polished web interface.

Key Features
Secure & Private: Runs 100% locally using Ollama and the phi3:mini model. No user data is sent to the cloud.

Fact-Grounded Responses: Implements an advanced RAG pipeline with a Parent Document Retriever to prevent AI "hallucination" and ensure all answers are based on the provided PDF documents.

Conversational Memory: Remembers the last few turns of the conversation to understand follow-up questions and provide a natural user experience.

Built-in Safety Protocols: Includes a robust, multi-layered system prompt to handle sensitive topics, redirect concerning statements, and provide immediate crisis intervention information.

Polished User Interface: A clean, responsive, and modern UI built with Streamlit, featuring a dark mode, custom chat bubbles, and suggested prompts for user guidance.

Fast & Efficient: Uses a FAISS vector store for rapid similarity searches and persists the knowledge base to disk for near-instant startups.

Architecture Overview
This project follows a classic Retrieval-Augmented Generation (RAG) architecture:

Data Ingestion: A PDF knowledge base (mental_health_guide.pdf) is loaded and processed by the pdf_loader.py script.

Indexing: The LangChain ParentDocumentRetriever splits the document into parent and child chunks. The child chunks are vectorized using OllamaEmbeddings and stored in a local FAISS vector store for efficient searching.

User Interface: The streamlit run app.py command launches a web interface where users can input queries.

Retrieval: When a user asks a question, the retriever finds the most relevant child chunks in the vector store and then retrieves their corresponding full-context parent chunks.

Generation: The user's question, the retrieved context, and the conversation history are passed to a locally-run phi3:mini model via Ollama. A carefully engineered system prompt guides the model to generate a safe, empathetic, and fact-grounded response.

Tech Stack
Web Framework: Streamlit

AI Orchestration: LangChain

LLM Serving: Ollama CLI

LLM Model: phi3:mini

Embeddings Model: OllamaEmbeddings

Vector Store: FAISS (Facebook AI Similarity Search)

PDF Processing: PyPDFLoader

Getting Started
Prerequisites
Python 3.9+

Ollama installed and running.

Installation & Setup
Clone the repository:

git clone [YOUR_GITHUB_REPO_LINK_HERE]
cd mental_health_bot

Create and activate a Python virtual environment:

# For Windows
python -m venv .venv
.venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

(Note: You may need to create a requirements.txt file by running pip freeze > requirements.txt)

Pull the Ollama model:

ollama pull phi3:mini

Place your knowledge base:

Add your mental_health_guide.pdf file to the root of the project directory.

Running the Application
Launch the Streamlit app:

streamlit run app.py

First-time setup: The application will automatically build the FAISS vector store and docstore from your PDF. This may take a moment. Subsequent launches will be near-instantaneous.

Open your browser: Navigate to http://localhost:8501 to start chatting with the AI.

Future Plans
Source Transparency: Implement an expander below each response to show the exact PDF chunks used for generation.

Database Integration: Add a database (e.g., SQLite or PostgreSQL) to store conversation histories for analytics and user feedback.

Cloud Deployment: Package the application in Docker and deploy it to a cloud service for persistent, multi-user access.
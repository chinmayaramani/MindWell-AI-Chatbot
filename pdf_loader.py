# pdf_loader.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the mental health PDF
loader = PyPDFLoader("mental_health_guide.pdf")
documents = loader.load()

# Split text into chunks for retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # ~1000 characters per chunk
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

print(f"[INFO] Loaded {len(docs)} chunks from PDF")

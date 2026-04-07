import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI(title="RAG Application")

# --------------- state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = None  # will hold the FAISS index after upload


# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT are allowed.")
    return loader.load()
    # TODO 


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store= FAISS.from_documents(chunks, embeddings)
    return store

    # TODO 


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    # TODO
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = store.as_retriever()
    #  `chain_type` → `"stuff"` (concatenates all retrieved docs into one prompt)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    # TODO 
    # - Define `@app.get("/", response_class=HTMLResponse)` → `async def home():- Return `Path("static/index.html").read_text()
    return Path("static/index.html").read_text()

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # TODO
    # - Define `@app.post("/upload")` → `async def upload_document(file: UploadFile = File(...)):`- Save the uploaded file to `UPLOAD_DIR` → Load the document using `load_document()` → Build the vector store using `build_vector_store()` → Store the vector store in the global variable `vector_store`
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    documents = load_document(str(file_path))
    global vector_store
    vector_store = build_vector_store(documents)
    return {"message": "File uploaded and processed successfully."} 


@app.post("/query")
async def query_document(req: QueryRequest):
    #TODO 
    # - Define `@app.post("/query")` → `async def query_document(req: QueryRequest):`- Check if `vector_store` is initialized → If not, raise an HTTPException with status code 400 → Get the QA chain using `get_qa_chain()` → Run the QA chain with the input question and return the answer
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")
    qa_chain = get_qa_chain(vector_store)
    answer = qa_chain.run(req.question)
    return {"answer": answer}


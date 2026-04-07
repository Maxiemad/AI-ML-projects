import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Application")

# --------------- config ---------------
UPLOAD_DIR = Path("uploads")
FAISS_DIR = Path("faiss_index")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
SCORE_THRESHOLD = 0.5
MAX_K = 10

UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = None  # FAISS index after upload / load

# query -> reuse identical (normalized + k) answers without calling the LLM again
query_cache: dict[tuple[str, int], dict] = {}


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


QA_PROMPT = PromptTemplate(
    template=(
        "Answer using only the context below. If the answer is not in the context, "
        "say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    input_variables=["context", "question"],
)


def load_document(file_path: str):
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(file_path).load()
    if path.suffix.lower() == ".txt":
        return TextLoader(file_path).load()
    raise ValueError("Unsupported file type. Only PDF and TXT are allowed.")


def build_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, get_embeddings())


def persist_store(store: FAISS):
    store.save_local(str(FAISS_DIR))


def make_retriever(store: FAISS, k: int):
    k = min(max(k, 1), MAX_K)
    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": SCORE_THRESHOLD},
    )


def get_qa_chain(store: FAISS, k: int):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = make_retriever(store, k)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )


def docs_to_sources(docs):
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


@app.on_event("startup")
def load_or_init_vector_store():
    global vector_store
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        vector_store = FAISS.load_local(
            str(FAISS_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )
        logger.info("Loaded FAISS index from %s", FAISS_DIR)


# --------------- routes ---------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("static/index.html").read_text()


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=3, ge=1, le=MAX_K)


@app.get("/documents")
async def list_documents():
    files = sorted(
        (f for f in UPLOAD_DIR.iterdir() if f.is_file()), key=lambda p: p.name
    )
    return {
        "files": [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
            for f in files
        ]
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    logger.info("Upload received: %s", file.filename)
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    documents = load_document(str(file_path))
    new_store = build_vector_store(documents)

    global vector_store
    if vector_store is None:
        vector_store = new_store
    else:
        vector_store.merge_from(new_store)

    persist_store(vector_store)
    query_cache.clear()
    logger.info("Vector store updated; index saved to %s", FAISS_DIR)

    return {"message": "File uploaded and processed successfully."}


@app.post("/query")
async def query_document(req: QueryRequest):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    cache_key = (req.question.strip().lower(), req.k)
    if cache_key in query_cache:
        logger.info("Cache hit for question (k=%s)", req.k)
        return query_cache[cache_key]

    retriever = make_retriever(vector_store, req.k)
    docs = retriever.invoke(req.question)
    if not docs:
        body = {
            "answer": "No relevant passages met the similarity threshold. Try rephrasing or lowering the threshold in code.",
            "sources": [],
        }
        logger.info("No chunks above threshold for question")
        return body

    qa_chain = get_qa_chain(vector_store, req.k)
    out = qa_chain.invoke({"query": req.question})
    body = {
        "answer": out["result"],
        "sources": docs_to_sources(out["source_documents"]),
    }
    query_cache[cache_key] = body
    logger.info("Answer returned for question (k=%s)", req.k)
    return body

# RAG Application – Suggested Upgrades

Each section below describes a small, self-contained improvement to the current pipeline.  
Upgrades are independent — implement any combination in any order.

---

## 1. Prompt Templating

**What to do:** Replace the default chain prompt with a custom `PromptTemplate` so you control how context and the question are presented to the LLM.

**Where to change:** `get_qa_chain()` in `main.py`

**What to implement:**
1. Import `PromptTemplate` from `langchain.prompts`.
2. Define a template string with `{context}` and `{question}` placeholders.
3. Pass it to `RetrievalQA.from_chain_type` via `chain_type_kwargs={"prompt": prompt}`.

**Hints:**
- Use `input_variables=["context", "question"]` to match the placeholders in your template string.
- Try instructions like "say I don't know if the answer isn't in the context" to reduce hallucinations.

**Why it matters:** The default prompt is generic. A custom template lets you add instructions (e.g. "answer in bullet points", "say I don't know if unsure") without changing any other logic.

**Documentation:**
- [PromptTemplate](https://python.langchain.com/docs/concepts/prompt_templates/)
- [RetrievalQA chain_type_kwargs](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)

---

## 2. Persist the FAISS Index

**What to do:** Save the vector store to disk after building it, and load it on startup if it already exists, so data survives restarts.

**Where to change:** `build_vector_store()` and app startup in `main.py`

**What to implement:**
1. After `FAISS.from_documents(...)`, call `store.save_local("faiss_index")`.
2. On startup, check if the `faiss_index/` directory exists and load it with `FAISS.load_local(...)`.

**Hints:**
- `FAISS.load_local` requires `allow_dangerous_deserialization=True` — only use with files you created yourself.
- Use `Path("faiss_index").exists()` to decide whether to load or build fresh.

**Why it matters:** Currently, uploading a document again after a restart means re-embedding from scratch. Persistence avoids that cost.

**Documentation:**
- [FAISS.save_local / load_local](https://python.langchain.com/docs/integrations/vectorstores/faiss/)

---

## 3. Return Source Chunks with Answers

**What to do:** Include the retrieved document chunks in the `/query` response so users can verify where the answer came from.

**Where to change:** `get_qa_chain()` and the `/query` route in `main.py`

**What to implement:**
1. Add `return_source_documents=True` to `RetrievalQA.from_chain_type(...)`.
2. Update the `/query` route to return both `"result"` and `"source_documents"`.

**Hints:**
- The chain response becomes a dict with keys `"result"` and `"source_documents"` instead of a plain string.
- Each item in `source_documents` is a LangChain `Document` with `.page_content` and `.metadata` (e.g. page number, filename).

**Why it matters:** Returning sources makes the system transparent and lets users spot hallucinations.

**Documentation:**
- [RetrievalQA return_source_documents](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)

---

## 4. Configurable Retrieval — Top-k

**What to do:** Expose `k` (the number of retrieved chunks) as a query parameter so callers can tune recall vs. speed per request.

**Where to change:** `QueryRequest` model and `get_qa_chain()` in `main.py`

**What to implement:**
1. Add an optional `k: int = 3` field to `QueryRequest`.
2. Pass it through to `store.as_retriever(search_kwargs={"k": k})`.

**Hints:**
- Add the `k` parameter to `get_qa_chain(store, k=3)` so the route can pass it in.
- Consider capping `k` at a reasonable maximum (e.g. 10) to prevent overly large prompts.

**Why it matters:** A low `k` is faster; a higher `k` captures more context for complex questions. Hard-coding it removes that flexibility.

**Documentation:**
- [as_retriever search_kwargs](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore.as_retriever)

---

## 5. Multi-document Support

**What to do:** Merge newly uploaded documents into the existing vector store instead of replacing it, so all previously uploaded files remain queryable.

**Where to change:** `/upload` route in `main.py`

**What to implement:**
1. Build the new store from the uploaded file as usual.
2. If `vector_store` is already populated, call `vector_store.merge_from(new_store)`.
3. Otherwise, assign it directly.

**Hints:**
- `merge_from` mutates the existing store in-place — no need to reassign.
- If you also implement upgrade 2 (persistence), re-save to disk after each merge.

**Why it matters:** The current pipeline silently discards previous uploads on every new upload.

**Documentation:**
- [FAISS.merge_from](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)

---

## 6. Conversation Memory

**What to do:** Replace `RetrievalQA` with `ConversationalRetrievalChain` and a `ConversationBufferMemory` so follow-up questions retain context from earlier turns.

**Where to change:** `get_qa_chain()` in `main.py`

**What to implement:**
1. Import `ConversationalRetrievalChain` and `ConversationBufferMemory`.
2. Create a shared `memory` instance (store it alongside `vector_store` in global state).
3. Build the chain using `ConversationalRetrievalChain.from_llm(...)`.

**Hints:**
- Set `memory_key="chat_history"` and `return_messages=True` on `ConversationBufferMemory`.
- The chain expects the input key `"question"` (not `"query"` like `RetrievalQA`).
- Reset memory by calling `memory.clear()` — useful to expose as a `DELETE /history` endpoint.

**Why it matters:** Without memory, every question is answered in isolation — "What did you just say?" would return nothing useful.

**Documentation:**
- [ConversationalRetrievalChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html)
- [ConversationBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer.ConversationBufferMemory.html)

---

## 7. Streaming Responses

**What to do:** Stream the LLM's output token-by-token to the client so the UI feels responsive instead of waiting for the full answer.

**Where to change:** `get_qa_chain()` and the `/query` route in `main.py`

**What to implement:**
1. Enable streaming on `ChatGroq` with `streaming=True`.
2. Use FastAPI's `StreamingResponse` with an async generator that yields tokens.

**Hints:**
- Use `AsyncIteratorCallbackHandler` from `langchain.callbacks` to collect tokens asynchronously.
- Wrap the chain call in `asyncio.create_task(chain.ainvoke(...))` so the generator and the chain run concurrently.
- Return `StreamingResponse(..., media_type="text/plain")` from the route.

**Why it matters:** For long answers, users see output immediately rather than staring at a spinner.

**Documentation:**
- [ChatGroq streaming](https://python.langchain.com/docs/integrations/chat/groq/)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)

---

## 8. Input Validation & File-type Guard

**What to do:** Reject unsupported file types at the upload boundary before they reach the document loader.

**Where to change:** `/upload` route in `main.py`

**What to implement:**
1. Define an `ALLOWED_EXTENSIONS` set at module level (e.g. `{".pdf", ".txt"}`).
2. Check `Path(file.filename).suffix.lower()` against it at the top of the `/upload` handler.
3. Raise `HTTPException(status_code=400, ...)` with a clear message if the check fails.

**Hints:**
- Do the check before writing anything to disk so no partial file is created.
- Include the allowed extensions in the error detail so the caller knows what to send instead.

**Documentation:**
- [FastAPI HTTPException](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [pathlib.Path.suffix](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.suffix)

---

## 9. Query Caching

**What to do:** Cache answers for repeated identical questions so the LLM is not called twice for the same input.

**Where to change:** `/query` route in `main.py`

**What to implement:**
1. Create a module-level `dict` (e.g. `query_cache: dict[str, str] = {}`).
2. Before invoking the chain, check if `req.question` is already a key in the cache.
3. If it is, return the cached answer. If not, run the chain and store the result.

**Hints:**
- Normalise the key (e.g. `req.question.strip().lower()`) so minor variations hit the same cache entry.
- The cache is in-memory — it resets on restart. Pair with upgrade 2 if you want persistence.
- For a production-grade cache, swap the dict for Redis using `redis-py`.

**Why it matters:** Repeated questions (common in demos and testing) currently cost a full LLM call each time.

**Documentation:**
- [Python dict](https://docs.python.org/3/library/stdtypes.html#dict)

---

## 10. `/documents` Endpoint — List Uploaded Files

**What to do:** Add a `GET /documents` route that returns the names of all files currently in the `uploads/` directory.

**Where to change:** `main.py` — add a new route after the existing ones

**What to implement:**
1. Use `Path("uploads").iterdir()` to list files.
2. Return a JSON list of filenames.
3. Optionally include file size and last-modified time from `file.stat()`.

**Hints:**
- Filter out directories with `if f.is_file()` in case subdirectories are ever created.
- Sort the list for a consistent response order, e.g. `sorted(..., key=lambda f: f.name)`.

**Why it matters:** There is currently no way for a caller to know which documents have been uploaded without checking the filesystem manually.

**Documentation:**
- [Path.iterdir](https://docs.python.org/3/library/pathlib.html#pathlib.Path.iterdir)
- [Path.stat](https://docs.python.org/3/library/pathlib.html#pathlib.Path.stat)

---

## 11. Similarity Score Threshold

**What to do:** Filter out retrieved chunks whose similarity score falls below a minimum threshold, so the LLM only sees genuinely relevant context.

**Where to change:** `get_qa_chain()` in `main.py`

**What to implement:**
1. Switch the retriever to `similarity_score_threshold` search type.
2. Set a `score_threshold` float (e.g. `0.5`) in `search_kwargs`.
3. If no chunks pass the threshold, return a fallback response instead of calling the LLM.

**Hints:**
- Use `store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})`.
- A threshold of `0.5` is a reasonable starting point — tune it by testing with your own documents.
- Catching the empty-retriever case prevents the LLM from hallucinating an answer with no context.

**Why it matters:** Without a threshold, the retriever always returns `k` chunks even if none are relevant to the question.

**Documentation:**
- [VectorStore.as_retriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore.as_retriever)

---

## 12. Structured Logging

**What to do:** Replace any print statements with structured log entries so you can trace requests, answers, and errors in a consistent format.

**Where to change:** `main.py` — add a logger at module level and emit log calls in routes

**What to implement:**
1. Import Python's `logging` module and configure it with `logging.basicConfig`.
2. Create a logger: `logger = logging.getLogger(__name__)`.
3. Log key events: file upload received, vector store built, question asked, answer returned, errors.

**Hints:**
- Use `logger.info(...)` for normal events and `logger.exception(...)` inside `except` blocks to capture the full traceback automatically.
- For JSON-formatted logs (useful with log aggregators), install and use `python-json-logger`.
- Avoid logging full document content — it can be very large and may leak sensitive data.

**Why it matters:** Without logging, debugging a failed upload or a bad answer requires guesswork.

**Documentation:**
- [Python logging](https://docs.python.org/3/library/logging.html)
- [python-json-logger](https://github.com/madzak/python-json-logger)

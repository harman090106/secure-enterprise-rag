# Secure Enterprise RAG with Role-Based Access Control (RBAC)

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline designed to securely query internal company documents. This project implements strict Role-Based Access Control at the vector database level to prevent cross-tenant data leakage and unauthorized access to confidential information.

## 🏗️ Architecture & Tech Stack
* **Orchestration:** LangChain (LCEL)
* **Vector Database:** Qdrant (Local instance)
* **LLM Inference:** Meta Llama-3 (8B) via Groq API for ultra-low latency.
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local CPU generation)
* **Observability:** LangSmith for trace logging and token monitoring.

## ✨ Core Features
1. **Metadata Pre-Filtering (RBAC):** Users are assigned clearance levels and department tags. The system dynamically constructs Qdrant filters to restrict the vector search space *before* context is passed to the LLM.
2. **Pre-Retrieval Guardrails:** An LLM-based classification router intercepts off-topic or external queries (e.g., coding help, general trivia) to save compute costs and prevent prompt injection.
3. **Conversational Memory:** Maintains state across queries, allowing the LLM to resolve pronouns and contextual follow-ups.
4. **Safety Alignment Bypass:** Utilizes simulation framing to allow heavily aligned open-weight models to process mock confidential data without triggering false-positive safety refusals.

## 🚀 How to Run Locally
1. Clone the repository and install dependencies: `pip install -r requirements.txt`
2. Add your Groq and LangSmith API keys to a `.env` file.
3. Run `python ingest_data.py` to initialize the Qdrant DB and embed the mock enterprise data.
4. Run `python secure_rag.py` to interact with the pipeline.
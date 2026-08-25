# SpecialistAI Assistant API (FastAPI)

A retrieval-augmented generation (RAG) API that answers clinical reference questions with grounded, source-cited responses, built for healthcare workers in rural and underserved clinics.

This is a Python rebuild of the retrieval and chat tier of [SpecialistAI-Assistant-Api](../SpecialistAI-Assistant-API), a .NET service that performs the same role. Both sit on top of the same ingestion pipeline ([SpecialistAI-Assistant-Function](../SpecialistAI-Assistant-Function)), the same Azure AI Search index, and the same Azure OpenAI deployments. This version reimplements the answering half in FastAPI with LangGraph handling orchestration.

## Why this exists

Clinicians in low-resource settings often lack easy access to searchable, current reference material. This API answers questions **only from a curated, ingested document set**, never from the model's general training knowledge, so every answer stays traceable back to real source documents.

The Python rebuild also served a second purpose: making the refusal decision explicit. In the .NET version, whether the assistant declines to answer emerges from prompt instructions and is detected after the fact by scanning the answer text for known phrases. Here it is a discrete graph node with its own structured output, which makes it inspectable and independently testable.

## Architecture


```
Client (message + conversation history)
  │
  ▼
POST /api/chat  (API key required)
  │
  ├─► InputGuardrailService: classifies the message (ALLOWED / OFF_TOPIC / HARMFUL)
  │                            via a lightweight LLM call, before any retrieval happens.
  │                            Blocked messages return a canned answer and never reach the graph.
  │
  └─► LangGraph state machine:
  │
  │      START ──► retrieve ──► grade ──┬──► answer ──► END
  │                                     └──► refuse ──► END
  │
  │      retrieve  embeds the question, runs hybrid search (keyword + vector) with
  │                semantic reranking against Azure AI Search
  │      grade     asks the model whether the retrieved passages can actually answer
  │                the question; returns a structured verdict plus a one-sentence reason
  │      answer    generates a cited response constrained to the retrieved passages
  │      refuse    returns a fixed message with no sources, no LLM call
  │
  └─► Response: answer + the document sections actually used
```

**Key design decisions:**

- **Grading as a separate node.** Retrieval always returns its top matches, even when nothing relevant exists. Rather than relying on the answering prompt to notice this, a dedicated node judges sufficiency before generation. The verdict and its reasoning are available on the `/chat/graph` route, which makes it possible to see *why* a request was refused rather than inferring it from the answer text.

- **Hybrid search with semantic reranking.** Search combines BM25 keyword matching with vector similarity, then reranks with Azure AI Search's semantic ranker. Reranker scores are a useful signal in practice: relevant queries against this corpus score around 3.0, off-corpus queries around 1.5.

- **Separate internal and external models.** Internal models (`SearchResponse`, `GraphChatResponse`) carry retrieval scores, grading verdicts, and other detail. The public contract (`ApiChatResponse`, `ApiChatSource`) exposes only what the client needs, in camelCase. This keeps the API surface stable when internal shapes change.

- **Defense in depth on content safety.** The guardrail classifier screens input before retrieval; Azure OpenAI's own content filter acts as an independent backstop on the completion call. A content-filtered classification is treated as harmful. Unparseable classifier output defaults to allowed, on the reasoning that a classifier hiccup should not block a legitimate clinical question.

- **Singleton clients via lifespan.** Search and OpenAI clients are constructed once at application startup and closed on shutdown, rather than per request, so connection pools are reused.

## Tech stack

| Concern | Technology |
|---|---|
| API framework | FastAPI |
| Orchestration | LangGraph (state machine) + LangChain (`AzureChatOpenAI`) |
| Chat model | Azure OpenAI, `gpt-5-mini` |
| Embeddings | Azure OpenAI, `text-embedding-3-large` (3072 dimensions) |
| Search | Azure AI Search, hybrid (keyword + vector) with semantic reranking |
| Config | pydantic-settings, `.env` locally / environment variables in Azure |
| API docs | OpenAPI, auto-generated at `/docs` |
| Auth | API key via a FastAPI dependency |
| Hosting | Azure Container Apps |

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Production endpoint. Matches the .NET API contract exactly, so the existing Angular UI can call either service without changes. |
| `POST` | `/chat/graph` | Same pipeline, but also returns the routing path taken and the grader's reasoning. Useful for debugging and for demonstrating the graph's behavior. |
| `GET` | `/search` | Raw retrieval results with scores. No LLM involvement. |
| `GET` | `/health` | Liveness check; echoes back the configured index and chat deployment. |

### `POST /api/chat`

Request:

```json
{
  "message": "What is the recommended treatment for uncomplicated malaria?",
  "history": []
}
```

Response:

```json
{
  "answer": "WHO recommends treating uncomplicated P. falciparum malaria with an artemisinin-based combination therapy (ACT) [1][4]...",
  "sources": [
    {
      "chunkId": "who-guidelines-for-malaria__artemisinin-based-combination-therapy__1",
      "documentId": "who-guidelines-for-malaria",
      "documentTitle": "13b23623-7edc-4092-8df4-9a31cc8c07ad.pdf",
      "sectionTitle": "Artemisinin-based combination therapy"
    }
  ]
}
```

Requires an `X-Api-Key` header. Missing or incorrect keys return `401`. Empty or over-length messages return `400`. Off-topic and harmful messages return `200` with a canned answer and no sources, so the client can render them as ordinary assistant replies.

## Getting started

### Prerequisites

- Python 3.12+
- An Azure AI Search resource with a populated index (see [SpecialistAI-Assistant-Function](../SpecialistAI-Assistant-Function) for ingestion)
- An Azure OpenAI resource with `gpt-5-mini` and `text-embedding-3-large` deployed

### Configuration

Create a `.env` in the project root (gitignored):

```
SEARCH_ENDPOINT=https://<your-search-resource>.search.windows.net
SEARCH_API_KEY=<key>
SEARCH_INDEX_NAME=chunks-index
SEARCH_SEMANTIC_CONFIG=chunks-semantic-config
SEARCH_VECTOR_FIELD=contentVector

OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com
OPENAI_API_KEY=<key>
OPENAI_API_VERSION=<api-version>
CHAT_DEPLOYMENT=gpt-5-mini
EMBEDDING_DEPLOYMENT=text-embedding-3-large

API_KEY=<key clients must send in X-Api-Key>
```

Missing values fail fast at startup with a Pydantic validation error naming the field, rather than surfacing as a connection error on the first request.

### Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

### Verify

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: <your key>" \
  -d '{"message": "What is the recommended treatment for uncomplicated malaria?", "history": []}'
```

To see the graph's routing decision on the same question, call `/chat/graph` instead. Asking something outside the corpus is the more interesting test: the response should come back with `"path": "refused"`, no sources, and a `grade_reason` explaining what the retrieved passages actually covered.

## Deployment

Containerized and deployed to Azure Container Apps. The image listens on port 8080. All `.env` values are supplied as environment variables; secrets are stored as Container App secrets rather than inline values.

```bash
docker build -t <registry>.azurecr.io/specialistai-fastapi:latest .
docker push <registry>.azurecr.io/specialistai-fastapi:latest
```

`.dockerignore` excludes `.env` and `.venv`. Without it, `COPY . .` would bake credentials into the image layers.

## Differences from the .NET implementation

Both services satisfy the same client contract, but they are not identical.

| | .NET (Semantic Kernel) | Python (LangGraph) |
|---|---|---|
| Retrieval | Agentic. The model decides whether and how many times to search. | One retrieval per request. |
| Query decomposition | Yes, via a plugin the model can call for multi-part questions. | Not implemented. |
| Refusal | Emerges from the system prompt; detected afterwards by phrase matching on the answer text. | An explicit graph node with a structured verdict and stated reasoning. |
| Conversation history | Sent to the model, capped to recent turns. | Accepted in the request, currently unused. |
| Source tracking | An invocation filter observes every tool call and dedupes across searches. | Sources come directly from the single retrieval. |

The agentic retrieval and decomposition in the .NET version handle genuinely multi-part questions better. The graph-based refusal in this version is more transparent and easier to test. Neither is strictly ahead of the other.

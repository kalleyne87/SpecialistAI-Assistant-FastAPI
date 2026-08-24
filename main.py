from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Request

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends

from config import Settings, get_settings
from graph import ChatGraph
from models import ApiChatRequest, ApiChatResponse, ApiChatSource, GraphChatResponse, SearchResponse, ChatRequest, ChatResponse
from search_service import SearchService
from chat_service import ChatService
from guardrail_service import InputGuardrailService

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.search_service = SearchService(settings)
    app.state.chat_service = ChatService(settings, app.state.search_service)
    app.state.chat_graph = ChatGraph(settings, app.state.search_service)
    app.state.guardrail = InputGuardrailService(settings)
    yield
    await app.state.search_service.close()


app = FastAPI(title="SpecialistAI Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {
        "status": "ok",
        "index": settings.search_index_name,
        "chat_deployment": settings.chat_deployment,
    }


@app.get("/search", response_model=list[SearchResponse])
async def search(q: str, settings: Settings = Depends(get_settings)):
    service = SearchService(settings)
    return await service.search(q)

@app.post("/api/chat", response_model=ApiChatResponse, response_model_by_alias=True)
async def api_chat(body: ApiChatRequest, request: Request):
    guardrail: InputGuardrailService = request.app.state.guardrail

    error = guardrail.initial_check(body.message)
    if error:
        raise HTTPException(status_code=400, detail=error)

    classification = await guardrail.classify(body.message)
    blocked = guardrail.blocked_answer(classification)
    if blocked:
        return ApiChatResponse(answer=blocked, sources=[])

    graph: ChatGraph = request.app.state.chat_graph
    result = await graph.ask(body.message)

    return ApiChatResponse(
        answer=result.answer,
        sources=[
            ApiChatSource(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                section_title=hit.section_title or "",
            )
            for hit in result.sources
        ],
    )

@app.post("/chat/graph", response_model=GraphChatResponse)
async def chat_graph(body: ChatRequest, request: Request):
    graph: ChatGraph = request.app.state.chat_graph
    return await graph.ask(body.question, top=body.top)

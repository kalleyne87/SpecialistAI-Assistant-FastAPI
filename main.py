from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends

from config import Settings, get_settings
from graph import ChatGraph
from models import GraphChatResponse, SearchResponse, ChatRequest, ChatResponse
from search_service import SearchService
from chat_service import ChatService

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.search_service = SearchService(settings)
    app.state.chat_service = ChatService(settings, app.state.search_service)
    app.state.chat_graph = ChatGraph(settings, app.state.search_service)
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

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    service: ChatService = request.app.state.chat_service
    return await service.ask(body.question, top=body.top)

@app.post("/chat/graph", response_model=GraphChatResponse)
async def chat_graph(body: ChatRequest, request: Request):
    graph: ChatGraph = request.app.state.chat_graph
    return await graph.ask(body.question, top=body.top)

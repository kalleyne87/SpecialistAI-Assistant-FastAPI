from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config import Settings, get_settings
from models import SearchHit, SearchResponse
from search_service import SearchService

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.search_service = SearchService(settings)
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
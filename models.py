from pydantic import BaseModel
from typing import Literal, TypedDict

class SearchResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    section_title: str | None = None
    sub_chunk_index: int | None = None
    text: str
    score: float
    reranker_score: float | None = None

class ChatRequest(BaseModel):
    question: str
    top: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchResponse]

class Grade(BaseModel):
    sufficient: bool
    reason: str

class GraphState(TypedDict):
    question: str
    top: int
    hits: list[SearchResponse]
    sufficient: bool
    grade_reason: str
    answer: str

class GraphChatResponse(BaseModel):
    answer: str
    sources: list[SearchResponse]
    path: Literal["answered", "refused"]
    grade_reason: str
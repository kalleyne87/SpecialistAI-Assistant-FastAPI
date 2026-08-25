from enum import Enum

from pydantic import BaseModel
from typing import Literal, TypedDict
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

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

class ConversationTurn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    role: str
    text: str


class ApiChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message: str
    history: list[ConversationTurn] = Field(default_factory=list)


class ApiChatSource(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    chunk_id: str
    document_id: str
    document_title: str
    section_title: str = ""


class ApiChatResponse(BaseModel):
    answer: str
    sources: list[ApiChatSource] = Field(default_factory=list)

class Classification(str, Enum):
    ALLOWED = "ALLOWED"
    OFF_TOPIC = "OFF_TOPIC"
    HARMFUL = "HARMFUL"
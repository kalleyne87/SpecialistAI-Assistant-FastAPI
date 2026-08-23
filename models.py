from pydantic import BaseModel


class SearchResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    section_title: str | None = None
    sub_chunk_index: int | None = None
    text: str
    score: float
    reranker_score: float | None = None
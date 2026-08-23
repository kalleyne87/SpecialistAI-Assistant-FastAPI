from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI

from config import Settings
from models import SearchResponse

class SearchService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = SearchClient(
            endpoint=settings.search_endpoint,
            index_name=settings.search_index_name,
            credential=AzureKeyCredential(settings.search_api_key),
        )
        self._openai = AsyncAzureOpenAI(
            azure_endpoint=settings.openai_endpoint,
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
        )

    async def embed(self, text: str) -> list[float]:
        result = await self._openai.embeddings.create(
            model=self._settings.embedding_deployment,
            input=text,
        )
        return result.data[0].embedding

    async def search(self, query: str, top: int = 5) -> list[dict]:
        vector = await self.embed(query)

        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top,
            fields=self._settings.search_vector_field,
        )

        results = await self._client.search(
            search_text=query,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name=self._settings.search_semantic_config,
            select=[
                "id",
                "documentId",
                "documentTitle",
                "sectionTitle",
                "subChunkIndex",
                "text",
            ],
            top=top,
        )

        return [
            SearchResponse(
                chunk_id=doc["id"],
                document_id=doc["documentId"],
                document_title=doc["documentTitle"],
                section_title=doc.get("sectionTitle"),
                sub_chunk_index=doc.get("subChunkIndex"),
                text=doc["text"],
                score=doc["@search.score"],
                reranker_score=doc.get("@search.rerankerScore"),
            )
            async for doc in results
        ]

    async def close(self) -> None:
        await self._client.close()
        await self._openai.close()
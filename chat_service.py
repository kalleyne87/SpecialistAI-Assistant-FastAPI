from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from config import Settings
from models import ChatResponse, SearchResponse
from search_service import SearchService

SYSTEM_PROMPT = """You are a clinical reference assistant. Answer the user's \
question using only the numbered context passages provided. Cite each passage \
you rely on in its own brackets, like [1] or [1][4]. Never combine numbers in \
one bracket, such as [1,4].

If the context does not contain enough information to answer, say so plainly \
and do not answer from your own knowledge."""

class ChatService:
    def __init__(self, settings: Settings, search_service: SearchService):
        self._search = search_service
        self._llm = AzureChatOpenAI(
            azure_endpoint=settings.openai_endpoint,
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            azure_deployment=settings.chat_deployment,
        )

    def _format_context(self, hits: list[SearchResponse]) -> str:
        blocks = []
        for i, hit in enumerate(hits, start=1):
            title = hit.section_title or hit.document_title
            blocks.append(f"[{i}] {title}\n{hit.text}")
        return "\n\n".join(blocks)

    async def ask(self, question: str, top: int = 5) -> ChatResponse:
        hits = await self._search.search(question, top=top)

        if not hits:
            return ChatResponse(
                answer="I could not find anything in the available documents "
                       "to answer that question.",
                sources=[],
            )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n\n{self._format_context(hits)}\n\n"
                        f"Question: {question}"
            ),
        ]

        result = await self._llm.ainvoke(messages)

        return ChatResponse(answer=result.content, sources=hits)
    
    
    
    
    


    
    
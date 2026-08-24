from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import Settings
from models import Grade, GraphChatResponse, GraphState, SearchResponse
from search_service import SearchService

GRADE_PROMPT = """You judge whether the provided context passages contain \
    enough information to answer the user's question.

    Answer with sufficient=true only if the passages directly address the question. \
    Passages that are merely from the same broad field are not enough. Give a one \
    sentence reason."""

ANSWER_PROMPT = """You are a clinical reference assistant. Answer the user's \
    question using only the numbered context passages provided. Cite each passage \
    you rely on in its own brackets, like [1] or [1][4]. Never combine numbers in \
    one bracket, such as [1,4]."""

def format_context(hits: list[SearchResponse]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        title = hit.section_title or hit.document_title
        blocks.append(f"[{i}] {title}\n{hit.text}")
    return "\n\n".join(blocks)

class ChatGraph:
    def __init__(self, settings: Settings, search_service: SearchService):
        self._search = search_service
        self._llm = AzureChatOpenAI(
            azure_endpoint=settings.openai_endpoint,
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            azure_deployment=settings.chat_deployment,
        )
        self._grader = self._llm.with_structured_output(Grade)
        self._graph = self._build()

    async def _retrieve(self, state: GraphState) -> dict:
        hits = await self._search.search(state["question"], top=state["top"])
        return {"hits": hits}

    async def _grade(self, state: GraphState) -> dict:
        if not state["hits"]:
            return {
                "sufficient": False,
                "grade_reason": "No passages were retrieved.",
            }

        result: Grade = await self._grader.ainvoke([
            SystemMessage(content=GRADE_PROMPT),
            HumanMessage(
                content=f"Context:\n\n{format_context(state['hits'])}\n\n"
                        f"Question: {state['question']}"
            ),
        ])
        return {"sufficient": result.sufficient, "grade_reason": result.reason}

    async def _answer(self, state: GraphState) -> dict:
        result = await self._llm.ainvoke([
            SystemMessage(content=ANSWER_PROMPT),
            HumanMessage(
                content=f"Context:\n\n{format_context(state['hits'])}\n\n"
                        f"Question: {state['question']}"
            ),
        ])
        return {"answer": result.content}

    async def _refuse(self, state: GraphState) -> dict:
        return {
            "answer": "I could not find enough information in the available "
                    "documents to answer that question.",
        }

    def _route(self, state: GraphState) -> Literal["answer", "refuse"]:
        return "answer" if state["sufficient"] else "refuse"

    def _build(self):
        builder = StateGraph(GraphState)

        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("answer", self._answer)
        builder.add_node("refuse", self._refuse)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges("grade", self._route)
        builder.add_edge("answer", END)
        builder.add_edge("refuse", END)

        return builder.compile()

    async def ask(self, question: str, top: int = 5) -> GraphChatResponse:
        final = await self._graph.ainvoke({"question": question, "top": top})

        sufficient = final["sufficient"]
        return GraphChatResponse(
            answer=final["answer"],
            sources=final["hits"] if sufficient else [],
            path="answered" if sufficient else "refused",
            grade_reason=final["grade_reason"],
        )
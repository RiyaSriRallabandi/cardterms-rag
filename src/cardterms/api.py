"""HTTP interface to the question-answering service.

The response carries the passages the answer was written from, not only the
answer. Error analysis found that a share of answers are confidently wrong with
a citation attached, so an interface that shows the answer alone would present
those failures as successes. The client renders the source text beside the
answer, and the API exists to make that possible.

    uv run uvicorn cardterms.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cardterms.logging import configure_logging, log
from cardterms.service import CardTerms

WEB_DIR = Path("web")

service: CardTerms | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once. The BM25 index, cross-encoder weights and
    entity vocabulary take tens of seconds and are reused by every request."""
    global service
    configure_logging(json_output=False)
    # The generator is chosen at startup rather than in code. Hosted models are
    # retired and rate limits are exhausted on the provider's schedule, not
    # ours, and swapping to the local fallback or a different hosted model
    # should not require editing a source file.
    overrides = {}
    if provider := os.getenv("CARDTERMS_PROVIDER"):
        overrides["provider"] = provider
    if model := os.getenv("CARDTERMS_MODEL"):
        overrides["model"] = model
    service = CardTerms(**overrides)
    yield
    service.close()


app = FastAPI(
    title="CardTerms",
    description="Grounded question answering over US credit card agreements.",
    version="1.0",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class PassageOut(BaseModel):
    number: int
    product: str
    issuer: str
    page: int
    text: str


class TimingOut(BaseModel):
    retrieve_ms: float
    rerank_ms: float
    generate_ms: float
    total_ms: float


class AnswerOut(BaseModel):
    question: str
    answer: str
    abstained: bool
    abstention_kind: str | None
    cited: list[int]
    entities: list[str]
    passages: list[PassageOut]
    timing: TimingOut


class CardOut(BaseModel):
    id: int
    issuer: str
    product: str


@app.post("/api/ask", response_model=AnswerOut)
def ask(payload: Question) -> AnswerOut:
    result = service.ask(payload.question.strip())
    log.info(
        "answered",
        abstained=result.abstained,
        total_ms=round(result.timing.total),
    )
    return AnswerOut(
        question=result.question,
        answer=result.answer,
        abstained=result.abstained,
        abstention_kind=result.abstention_kind,
        cited=result.cited,
        entities=result.entities,
        passages=[
            PassageOut(
                number=p.number,
                product=p.product,
                issuer=p.issuer,
                page=p.page,
                text=p.text,
            )
            for p in result.passages
        ],
        timing=TimingOut(
            retrieve_ms=round(result.timing.retrieve, 1),
            rerank_ms=round(result.timing.rerank, 1),
            generate_ms=round(result.timing.generate, 1),
            total_ms=round(result.timing.total, 1),
        ),
    )


@app.get("/api/cards", response_model=list[CardOut])
def cards(q: str = "", limit: int = 40) -> list[CardOut]:
    """Cards matching a substring — used when a question names none."""
    return [CardOut(**row) for row in service.products(q.strip(), limit)]


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": service is not None}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.bias_detector import BiasDetectorService


router = APIRouter()


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze for bias")


class BiasedSpan(BaseModel):
    text: str
    start: int
    end: int
    type: str


class SentenceAnalysis(BaseModel):
    sentence: str
    biased_spans: List[BiasedSpan]
    suggestion: str


class AnalyzeResponse(BaseModel):
    original_text: str
    summary: Dict[str, Any]
    sentences: List[SentenceAnalysis]


detector = BiasDetectorService.create_default()


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return detector.analyze_text(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



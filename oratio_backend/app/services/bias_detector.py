from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import spacy
from pydantic import BaseModel

from ..core.config import settings
from ..models.tf_model import TensorflowBiasScorer
from ..models.torch_model import TorchBiasScorer


class BiasDetectorService:
    def __init__(
        self,
        nlp: spacy.language.Language,
        tf_model: TensorflowBiasScorer,
        torch_model: TorchBiasScorer,
    ) -> None:
        self._nlp = nlp
        self._tf = tf_model
        self._torch = torch_model

    @classmethod
    def create_default(cls) -> "BiasDetectorService":
        nlp = spacy.load(settings.spacy_model, disable=["ner", "lemmatizer"])  # speed
        tf_model = TensorflowBiasScorer()
        torch_model = TorchBiasScorer(use_gpu=settings.use_gpu)
        return cls(nlp=nlp, tf_model=tf_model, torch_model=torch_model)

    def analyze_text(self, text: str) -> "AnalyzeResponse":
        if not text or not text.strip():
            raise ValueError("Text is empty")

        doc = self._nlp(text)

        sentences: List[SentenceAnalysis] = []
        biased_total = 0
        scores: List[float] = []

        for sent in doc.sents:
            spans = self._find_biased_spans(sent.text)
            biased_total += len(spans)

            # Score with both models and average
            tf_score = float(self._tf.score(sent.text))
            torch_score = float(self._torch.score(sent.text))
            score = float((tf_score + torch_score) / 2.0)
            scores.append(score)

            suggestion = self._suggest_rewrite(sent.text, spans)
            sentences.append(
                SentenceAnalysis(
                    sentence=sent.text,
                    biased_spans=[
                        BiasedSpan(text=s.text, start=s.start, end=s.end, type=s.type)
                        for s in spans
                    ],
                    suggestion=suggestion,
                )
            )

        summary = {"biased_count": biased_total, "score": float(np.mean(scores) if scores else 0.0)}
        return AnalyzeResponse(original_text=text, summary=summary, sentences=sentences)

    def _find_biased_spans(self, sentence: str) -> List["SpanResult"]:
        # Simple lexical triggers as a placeholder; replace with trained models later
        lexicon = {
            "ableist": {"crazy", "insane", "lame"},
            "ageist": {"old", "boomer"},
            "sexist": {"bossy", "hysterical"},
            "racist": {"illegal", "exotic"},
        }

        spans: List[SpanResult] = []
        lower = sentence.lower()
        cursor = 0
        for token in sentence.split():
            start = lower.find(token.lower(), cursor)
            end = start + len(token)
            cursor = end
            for bias_type, words in lexicon.items():
                if token.lower().strip(".,!?\"'()") in words:
                    spans.append(SpanResult(text=sentence[start:end], start=start, end=end, type=bias_type))
        return spans

    def _suggest_rewrite(self, sentence: str, spans: List["SpanResult"]) -> str:
        if not spans:
            return sentence
        suggestion = sentence
        replacements = {
            "crazy": "unusual",
            "insane": "extreme",
            "lame": "uninspired",
            "old": "older",
            "boomer": "older person",
            "bossy": "directive",
            "hysterical": "overwhelmed",
            "illegal": "undocumented",
            "exotic": "distinct",
        }
        for span in spans:
            key = span.text.lower()
            clean_key = key.strip(".,!?\"'()")
            if clean_key in replacements:
                suggestion = suggestion.replace(span.text, replacements[clean_key])
        return suggestion


class SpanResult(BaseModel):
    text: str
    start: int
    end: int
    type: str


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



"""Placeholder interfaces for future ArchAI pipeline agents."""

from __future__ import annotations


class RetrievalAgent:
    def index(self, *args, **kwargs) -> None:  # pragma: no cover
        return None


class KnowledgeExtractionAgent:
    def extract(self, *args, **kwargs) -> None:  # pragma: no cover
        return None


class AuthorityLinkAgent:
    def link(self, *args, **kwargs) -> None:  # pragma: no cover
        return None


class RagAgent:
    def retrieve(self, *args, **kwargs) -> None:  # pragma: no cover
        return None


class AnswerAgent:
    def answer(self, *args, **kwargs) -> None:  # pragma: no cover
        return None


class VerificationAgent:
    def verify(self, *args, **kwargs) -> None:  # pragma: no cover
        return None

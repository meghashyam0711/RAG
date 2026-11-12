"""Typed data contracts shared by ingestion, retrieval and the agent."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constrained ontology. Extraction is never open-ended: the LLM (or the rule
# extractor) may only emit these types, which prevents schema drift.
# ---------------------------------------------------------------------------

ENTITY_TYPES: dict[str, list[str]] = {
    "general": ["Organization", "Person", "Component", "Parameter", "Concept", "Section", "Regulation", "FiscalPeriod"],
    "technical": ["Component", "Parameter", "Endpoint", "Protocol", "Section", "Concept", "Organization"],
    "finance": ["Organization", "FinancialMetric", "FiscalPeriod", "RiskFactor", "Executive", "Section", "Segment"],
    "legal": ["Party", "Clause", "Obligation", "Term", "Jurisdiction", "Section", "Regulation"],
}

RELATION_TYPES: list[str] = [
    "DEFINES",       # section/doc defines a parameter or term
    "MODIFIES",      # a later section overrides / amends something
    "REFERENCES",    # explicit cross-reference ("see Appendix B")
    "DEPENDS_ON",
    "PART_OF",
    "MENTIONS",
    "HAS_VALUE",
    "REPORTED_IN",
    "ACQUIRED",
    "MITIGATES",
]


class BlockType(str, Enum):
    heading = "heading"
    text = "text"
    table = "table"


class Block(BaseModel):
    """A layout element in reading order, produced by the layout parser."""

    type: BlockType
    text: str
    page: int | None = None
    section: str = ""
    units: str | None = None  # e.g. "$ in millions", attached to tables


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    domain: str
    text: str                       # raw content (what is shown to the caller)
    header: str                     # contextual metadata header prepended before embedding
    chunk_type: Literal["text", "table"] = "text"
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    units: str | None = None
    entity_ids: list[str] = Field(default_factory=list)

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"

    @property
    def citation(self) -> str:
        if self.page_start is None:
            pages = ""
        elif self.page_end and self.page_end != self.page_start:
            pages = f", pp. {self.page_start}-{self.page_end}"
        else:
            pages = f", p. {self.page_start}"
        sec = f", {self.section}" if self.section else ""
        return f"{self.doc_name}{sec}{pages}"


class ExtractedEntity(BaseModel):
    name: str = Field(description="Surface form exactly as written in the text")
    type: str = Field(description="One of the allowed entity types")
    description: str = ""


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str = Field(description="One of the allowed relation types")
    evidence: str = Field(default="", description="Short verbatim span supporting the relation")


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class Entity(BaseModel):
    """Canonical graph node after entity resolution."""

    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    description: str = ""


class Relation(BaseModel):
    source_id: str
    target_id: str
    type: str
    chunk_id: str
    evidence: str = ""


class Evidence(BaseModel):
    chunk: Chunk
    score: float = 0.0
    sources: list[str] = Field(default_factory=list)  # "dense", "sparse", "graph"
    graph_path: str | None = None


class GradeLabel(str, Enum):
    relevant = "RELEVANT"
    ambiguous = "AMBIGUOUS"
    missing = "MISSING"


class VerifyLabel(str, Enum):
    grounded = "GROUNDED"
    hallucinated = "HALLUCINATED"
    sign_error = "SIGN_ERROR"


class VerifiedAnswer(BaseModel):
    status: Literal["verified", "partially_verified", "unverified", "abstained"]
    answer: str
    citations: list[dict] = Field(default_factory=list)
    calculations: list[dict] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 0

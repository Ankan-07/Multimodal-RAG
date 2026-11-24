"""NEW: Record types used by the hybrid ingestion and storage path.

A record is one searchable entry. Text records go to the MiniLM collection.
Visual records go to the Gemini collection. The dataclasses keep the two
paths explicit and avoid passing loosely structured dictionaries everywhere.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextRecord:
    """A text chunk that will be embedded by the Sentence Transformer."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualRecord:
    """An image or table that will be embedded by Gemini Embedding 2."""

    id: str
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    kind: str = "image"
    # image_bytes is used for image records. embedding_text is used for tables.
    image_bytes: Optional[bytes] = None
    mime_type: Optional[str] = None
    embedding_text: Optional[str] = None


@dataclass
class IngestionBatch:
    """All records produced from one source document."""

    text_records: List[TextRecord] = field(default_factory=list)
    visual_records: List[VisualRecord] = field(default_factory=list)

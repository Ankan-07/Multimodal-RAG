"""BM25 keyword index for hybrid retrieval.

This module provides one small class, :class:`BM25Index`, that indexes text
chunks and normalized table text with ``rank_bm25.BM25L``.  A retriever can
merge the keyword results from this module with vector results (for example,
with reciprocal rank fusion).

The module imports only the standard library, ``rank_bm25``, and the project
``config`` module.  It does not import Chroma or any other project module.
"""

from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25L

import config

# A token is either a Unicode letter sequence (no digits, no underscore) or a
# number.  The number pattern keeps an optional decimal part, so "3.14" and
# "42" stay single tokens.  Everything else (punctuation, symbols, emoji) is a
# separator.
#
# Examples:
#   "Revenue 3.14"  -> ["revenue", "3.14"]
#   "v1.5"          -> ["v", "1.5"]
#   "GPT-4"         -> ["gpt", "4"]
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?")

# A comma between digit groups of exactly three digits is a thousands
# separator.  Removing it makes "1,234.56" and "1234.56" produce the same
# token, which helps when a table and a question format numbers differently.
_THOUSANDS_SEPARATOR_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word and number tokens.

    The function lowercases the text, folds accented characters to their base
    form (so "café" matches "cafe"), removes thousands separators, and drops
    all punctuation.  Decimal numbers and integers are preserved.
    """

    if not text:
        return []

    # Lowercase first, then decompose characters so combining accent marks can
    # be removed.  NFKD also converts full-width digits to ASCII digits.
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )

    # Normalize numeric formatting before token extraction.
    normalized = _THOUSANDS_SEPARATOR_RE.sub("", normalized)

    return _TOKEN_RE.findall(normalized)


class BM25Index:
    """An in-memory BM25 index over chunk dictionaries.

    Each input document must have this shape::

        {
            "id": str,
            "document": str,   # text or normalized table text
            "metadata": dict,  # source, page, chunk_id, kind, original_content
            "source": str,     # "text" or "table"
        }

    ``k1`` and ``b`` default to ``config.BM25_K1`` and ``config.BM25_B``.
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        # Store the BM25 parameters as floats so callers can pass integers.
        self.k1: float = float(config.BM25_K1 if k1 is None else k1)
        self.b: float = float(config.BM25_B if b is None else b)

        # All documents stay in input order.  A separate token list keeps the
        # tokenized text close to its document.
        self._documents: list[dict] = []
        self._tokens: list[list[str]] = []

        # BM25L is built only when at least one document has tokens.
        self._bm25: BM25L | None = None

        # Position of each BM25 corpus row in self._documents.  Documents with
        # no tokens are not indexed, so this list maps scores back correctly.
        self._indexed_positions: list[int] = []

        if documents:
            self.add_documents(documents)

    @property
    def size(self) -> int:
        """Return the number of documents that were added to the index."""

        return len(self._documents)

    def add_documents(self, documents: list[dict]) -> None:
        """Add documents and rebuild the BM25 index.

        A rebuild touches all stored tokens, so it costs O(total tokens).  This
        is acceptable for the ingest-time or start-up use of this class.
        """

        prepared = [
            self._prepare_document(document, position)
            for position, document in enumerate(documents)
        ]
        if not prepared:
            return

        self._documents.extend(prepared)
        self._tokens.extend(tokenize(document["document"]) for document in prepared)
        self._rebuild()

    def search(self, query: str, top_k: int) -> list[dict]:
        """Return up to ``top_k`` documents with a positive BM25 score.

        The result list is sorted by score in descending order.  An empty
        index, an empty token list, a non-positive ``top_k``, or a result set
        with no positive scores returns ``[]``.
        """

        if top_k <= 0 or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # BM25L returns one score per indexed (non-empty) document.
        scores = self._bm25.get_scores(query_tokens)

        # Sort by descending score and use the corpus position as a stable
        # tie-breaker, so equal scores always return in a predictable order.
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )

        results: list[dict] = []
        for bm25_position, score in ranked:
            # Scores are sorted, so the first non-positive score ends the
            # loop.  This keeps zero-score documents out of the result list.
            score_value = float(score)
            if score_value <= 0:
                break

            document = self._documents[self._indexed_positions[bm25_position]]
            results.append(
                {
                    "id": document["id"],
                    "document": document["document"],
                    # Copy the outer metadata dict so callers cannot change the
                    # stored document by editing the result.
                    "metadata": dict(document["metadata"]),
                    "score": score_value,
                    # The result source identifies the retrieval method, not
                    # the original document source ("text" or "table").
                    "source": "bm25",
                }
            )
            if len(results) >= top_k:
                break

        return results

    def _rebuild(self) -> None:
        """Build BM25L from the current token lists."""

        # BM25L divides by the corpus size and by the total token count.
        # An empty corpus or an all-empty corpus would raise ZeroDivisionError,
        # so skip those cases.
        self._indexed_positions = [
            position for position, tokens in enumerate(self._tokens) if tokens
        ]
        if not self._indexed_positions:
            self._bm25 = None
            return

        corpus = [self._tokens[position] for position in self._indexed_positions]
        # BM25L is more stable than BM25Okapi on very small corpora and short
        # table text. It keeps a positive score for a term that appears in only
        # one of two documents.
        self._bm25 = BM25L(corpus, k1=self.k1, b=self.b)

    @staticmethod
    def _prepare_document(document: dict, position: int) -> dict:
        """Validate one input document and return a normalized copy."""

        if not isinstance(document, dict):
            raise TypeError(f"document at position {position} must be a dict")
        if "id" not in document or "document" not in document:
            raise ValueError(
                f"document at position {position} needs 'id' and 'document' keys"
            )

        metadata = document.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise TypeError(
                f"metadata at position {position} must be a dict"
            )

        return {
            "id": str(document["id"]),
            "document": str(document["document"]),
            "metadata": dict(metadata),
            "source": str(document.get("source", "text")),
        }

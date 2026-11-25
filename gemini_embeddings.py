"""Gemini Embedding 2 client for image and table vectors.

Text chunks stay on the local MiniLM model. This module covers only the
visual collection: images and table text that must share one vector space.
The Gemini client is created on first use, so importing this module does not
open a network connection or require an API key.
"""

import io
import logging
import time
from typing import Any, List

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

import config

logger = logging.getLogger(__name__)

# Gemini Embedding 2 accepts these image formats directly. Any other image
# format is converted to PNG before the request.
SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}

# HTTP status codes that can succeed when the request is sent again.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Safety cap for text sent to the embedding API. The Gemini Developer API no
# longer accepts the auto_truncate flag (Enterprise-only), so oversized input
# is truncated locally to stay under the model context limit.
MAX_TEXT_CHARS_FOR_EMBEDDING = 10000

# Base delay for exponential backoff, and a cap so a high retry count does not
# stall a long ingestion run.
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 16.0


class GeminiEmbedder:
    """Create embeddings for text, tables, and images with Gemini Embedding 2."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        output_dim: int | None = None,
    ) -> None:
        # Resolve settings from config when the caller does not override them.
        self._api_key = api_key or config.GEMINI_API_KEY
        self._model = model or config.GEMINI_EMBEDDING_MODEL
        self._output_dim = (
            output_dim if output_dim is not None else config.GEMINI_EMBEDDING_DIM
        )
        self._request_retries = max(0, int(config.GEMINI_REQUEST_RETRIES))

        # Fail early on a missing key. This is a configuration error, not a
        # transient failure, so it must never be retried or hidden.
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY must be set in .env to embed images and tables "
                "with Gemini. GOOGLE_API_KEY is accepted as a fallback."
            )

        # The client is built lazily in _get_client, not here and not at import.
        self._client: genai.Client | None = None

    def embed_text(self, text: str) -> List[float]:
        """Embed one text string, such as normalized table text or a query."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        # Truncate locally instead of relying on the API auto_truncate flag
        # (which is unsupported in Gemini Developer API mode).
        if len(text) > MAX_TEXT_CHARS_FOR_EMBEDDING:
            logger.warning(
                "Truncating text from %d to %d characters for embedding.",
                len(text),
                MAX_TEXT_CHARS_FOR_EMBEDDING,
            )
            text = text[:MAX_TEXT_CHARS_FOR_EMBEDDING]

        # A plain string is a valid content value for the Gemini API.
        return self._embed_contents(text, "text")

    def embed_image_bytes(
        self, image_bytes: bytes, mime_type: str = "image/png"
    ) -> List[float]:
        """Embed one image, converting the bytes to PNG when necessary."""
        if not image_bytes:
            raise ValueError("Cannot embed empty image bytes.")

        # Drop parameters such as "; charset=..." and normalize case so the
        # format check is reliable.
        normalized_mime_type = (mime_type or "").split(";")[0].strip().lower()

        # Convert any unsupported image format instead of sending it silently.
        if normalized_mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            image_bytes, normalized_mime_type = _convert_to_png(image_bytes)

        part = types.Part.from_bytes(data=image_bytes, mime_type=normalized_mime_type)
        return self._embed_contents([part], f"image ({normalized_mime_type})")

    def _get_client(self) -> genai.Client:
        """Create the Gemini client on first use."""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _embed_contents(self, contents: Any, description: str) -> List[float]:
        """Send one embedding request, retrying only transient failures."""
        client = self._get_client()
        # The Gemini Developer API rejects auto_truncate (Enterprise-only), so
        # oversized input is truncated in embed_text. Images are resized by the
        # API or converted to PNG, and stay within the API limits.
        request_config = types.EmbedContentConfig(
            output_dimensionality=self._output_dim,
        )

        # One initial attempt plus GEMINI_REQUEST_RETRIES retries.
        for attempt in range(self._request_retries + 1):
            try:
                result = client.models.embed_content(
                    model=self._model,
                    contents=contents,
                    config=request_config,
                )
                break
            except Exception as exc:
                # Permanent errors, such as a bad key or an invalid model, are
                # re-raised on the first attempt.
                if not _is_retryable(exc) or attempt >= self._request_retries:
                    raise

                # Exponential backoff: 1 s, 2 s, 4 s, ... up to the cap.
                delay = min(
                    RETRY_BASE_DELAY_SECONDS * (2**attempt),
                    RETRY_MAX_DELAY_SECONDS,
                )
                logger.warning(
                    "Gemini embedding request for %s failed on attempt %d of %d: "
                    "%s. Retrying in %.1f seconds.",
                    description,
                    attempt + 1,
                    self._request_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        return _extract_vector(result, self._output_dim, description)


def _is_retryable(exc: Exception) -> bool:
    """Return True for rate limits, server errors, and network failures."""
    if isinstance(exc, genai_errors.APIError):
        return exc.code in RETRYABLE_STATUS_CODES

    # Timeouts and connection resets have no HTTP status code.
    return isinstance(
        exc,
        (httpx.TimeoutException, httpx.TransportError, ConnectionError, TimeoutError),
    )


def _convert_to_png(image_bytes: bytes) -> tuple[bytes, str]:
    """Convert image bytes to PNG with Pillow and return the new MIME type."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            # Keep transparency when the source has an alpha channel.
            if "A" in image.getbands():
                converted = image.convert("RGBA")
            else:
                converted = image.convert("RGB")

            output = io.BytesIO()
            converted.save(output, format="PNG")
    except Exception as exc:
        # A corrupt or non-image payload is a caller error, not a retry case.
        raise ValueError(f"Could not convert image data to PNG: {exc}") from exc

    return output.getvalue(), "image/png"


def _extract_vector(
    result: Any, output_dim: int | None, description: str
) -> List[float]:
    """Validate the API response and return its first embedding as floats."""
    embeddings = getattr(result, "embeddings", None)
    if not embeddings:
        raise RuntimeError(f"Gemini returned no embeddings for {description}.")

    values = getattr(embeddings[0], "values", None)
    if not values:
        raise RuntimeError(f"Gemini returned an empty embedding for {description}.")

    vector = [float(value) for value in values]

    # The visual collection has a fixed width, so a mismatch must fail loudly.
    if output_dim is not None and len(vector) != output_dim:
        raise RuntimeError(
            f"Gemini returned a {len(vector)}-dimension embedding for "
            f"{description}, but {output_dim} were requested."
        )

    return vector

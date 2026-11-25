# document_processor.py

This module turns a raw PDF into structured, chunked content for downstream
processing.

`partition_document` uses `unstructured` with `strategy="hi_res"` (layout
detection over each page), `infer_table_structure=True` (tables extracted as
HTML), and `extract_image_block_types=["Image"]` with
`extract_image_block_to_payload=True` (images kept as base64 in element
metadata, not written to disk).

`create_chunks_by_title` uses unstructured's title-aware chunker with the
config sizes: `max_characters=3000`, `new_after_n_chars=2500`,
`combine_text_under_n_chars=500`.

`separate_content_types` inspects each chunk's `orig_elements` and returns a
dict with `text`, `tables` (HTML strings), `images` (base64), `image_mime_types`, `types`, and `page_number`. Tables fall back to element
text when `text_as_html` is missing. Images keep their MIME type but fall
back to `image/jpeg`. Downstream, text becomes a `TextRecord`; tables and
images become `VisualRecord`s — no vision-LLM summarization is applied here.

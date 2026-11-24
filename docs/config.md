# config.py

This file is the single source of truth for every configurable value in the system. The idea here is that nothing outside this file should ever have a hardcoded string for a model name, a file path, or a numeric hyperparameter. If something needs to change — say we want to swap the summarizer model or adjust chunk sizes — you change it in exactly one place and it propagates everywhere automatically.

We load the `.env` file here using `python-dotenv`, so the `OPENAI_API_KEY` (which is actually the OpenRouter key, just named to satisfy LangChain's default expectation) gets picked up from the environment. This means in production you set it as an environment variable, in development you use a `.env` file, and the code never has to know the difference.

The `OPENROUTER_BASE_URL` override is what lets us point LangChain's `ChatOpenAI` client at OpenRouter instead of the real OpenAI. LangChain's OpenAI-compatible client accepts a `base_url` parameter for exactly this reason, and since OpenRouter exposes an OpenAI-compatible API, we get access to dozens of free and paid models without changing any client code.

We have two different model constants — `SUMMARIZER_MODEL` and `ANSWER_MODEL` — because these two tasks have different requirements. The summarizer needs to handle images (it's a vision-language model), while the answer generation model just needs to be good at reasoning over text. Keeping them separate means we can independently tune cost versus quality for each stage.

The chunking constants (`CHUNK_MAX_CHARACTERS`, `CHUNK_NEW_AFTER_N_CHARS`, `CHUNK_COMBINE_UNDER_N_CHARS`) control how aggressively we split or merge document sections. These values were tuned to balance context length versus retrieval granularity — chunks large enough to be meaningful, small enough for precise retrieval.

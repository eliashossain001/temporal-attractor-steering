"""Environment loading for HF auth tokens and cache paths.

Entry points should call `load_env()` once at startup before any
`load_model()` call; `get_hf_token()` then returns the token to pass
through to `transformers.AutoModelForCausalLM.from_pretrained(..., token=...)`.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import find_dotenv as _find_dotenv
    from dotenv import load_dotenv as _load_dotenv
except ImportError as e:
    raise ImportError(
        "python-dotenv is required. Install with: pip install python-dotenv"
    ) from e


def load_env(path: str | Path | None = None) -> Path | None:
    """Load a `.env` file into `os.environ`.

    If `path` is given, only that file is tried. Otherwise we walk up from
    the current working directory looking for a `.env`. Returns the path
    that was loaded, or None if no file was found.
    """
    if path is not None:
        p = Path(path)
        if p.exists():
            _load_dotenv(p, override=False)
            return p
        return None
    found = _find_dotenv(usecwd=True)
    if not found:
        return None
    _load_dotenv(found, override=False)
    return Path(found)


def get_hf_token() -> str | None:
    """Return the HF token from any conventional env var, or None."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v
    return None

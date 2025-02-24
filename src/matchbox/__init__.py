from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

# Environment variables must be loaded first for other imports to work
from matchbox.client import (  # noqa: E402
    match,
    query,
)

__all__ = ("query", "match")

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

# Environment variables must be loaded first for other imports to work

from matchbox.client.helpers.cleaner import process  # NoQA: E402
from matchbox.client.helpers.index import index  # NoQA: E402
from matchbox.client.helpers.selector import match, query, select  # NoQA: E402
from matchbox.client.models.models import make_model  # NoQA: E402

__all__ = ("make_model", "process", "select", "query", "match", "index")

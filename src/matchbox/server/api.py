from enum import StrEnum
from typing import Annotated

from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from matchbox.server.base import BackendManager, MatchboxDBAdapter

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


app = FastAPI(
    title="matchbox API",
    version="0.2.0",
)


class BackendEntityType(StrEnum):
    DATASETS = "datasets"
    MODELS = "models"
    DATA = "data"
    CLUSTERS = "clusters"
    CREATES = "creates"
    MERGES = "merges"
    PROPOSES = "proposes"


class ModelResultsType(StrEnum):
    PROBABILITIES = "probabilities"
    CLUSTERS = "clusters"


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class CountResult(BaseModel):
    """Response model for count results"""

    entities: dict[BackendEntityType, int]


def get_backend() -> MatchboxDBAdapter:
    return BackendManager.get_backend()


@app.get("/health")
async def healthcheck() -> HealthCheck:
    """ """
    return HealthCheck(status="OK")


@app.get("/testing/count")
async def count_backend_items(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    entity: BackendEntityType | None = None,
) -> CountResult:
    def get_count(e: BackendEntityType) -> int:
        return getattr(backend, str(e)).count()

    if entity is not None:
        return CountResult(entities={str(entity): get_count(entity)})
    else:
        res = {str(e): get_count(e) for e in BackendEntityType}
        return CountResult(entities=res)


@app.post("/testing/clear")
async def clear_backend():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources")
async def list_sources():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources/{hash}")
async def get_source(hash: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/sources/{hash}")
async def add_source(hash: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models")
async def list_models():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}")
async def get_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}")
async def add_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.delete("/models/{name}")
async def delete_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/results")
async def get_results(name: str, result_type: ModelResultsType | None):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/results")
async def set_results(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/truth")
async def get_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/truth")
async def set_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors")
async def get_ancestors(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors_cache")
async def get_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/ancestors_cache")
async def set_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/query")
async def query():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/match")
async def match():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/validate/hash")
async def validate_hashes():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/report/models")
async def get_model_subgraph():
    raise HTTPException(status_code=501, detail="Not implemented")

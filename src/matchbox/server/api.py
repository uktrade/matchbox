from binascii import hexlify
from enum import StrEnum
from typing import Annotated, Optional

from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile, Form
from pydantic import BaseModel
from matchbox.server.utils.aws_s3_client import upload_to_s3
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


class SourceItem(BaseModel):
    """Response model for source"""
    schema: str
    table: str
    id: str
    model: Optional[str] = None


class Sources(BaseModel):
    """Response model for sources"""
    sources: list[SourceItem]


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
async def list_sources(
        backend: Annotated[MatchboxDBAdapter, Depends(get_backend)]
) -> Sources:
    datasets = backend.datasets.list()
    result = []
    for dataset in datasets:
        print(dataset)
        result.append(SourceItem(
            table=getattr(dataset, "table"),
            id=getattr(dataset, "id"),
            schema=getattr(dataset, "schema"),
            model = hexlify(getattr(dataset, "model")).decode('ascii')
        ))
    return Sources(sources=result)


@app.get("/sources/{hash}")
async def get_source(hash: str,
                     backend: Annotated[MatchboxDBAdapter, Depends(get_backend)]
                     )-> dict[str, SourceItem] | str:
    datasets = backend.datasets.list()
    for dataset in datasets:
        model = hexlify(getattr(dataset, "model")).decode('ascii')
        if model == hash:
            result_obj = SourceItem(
            table=getattr(dataset, "table"),
            id=getattr(dataset, "id"),
            schema=getattr(dataset, "schema"),
            model = model)
            return {"source": result_obj}
    return "Source not found"


@app.post("/sources/uploadFile")
async def add_source_to_s3(file: UploadFile, bucket_name: str = Form(...), object_name: str = Form(...)):
    is_file_uploaded = upload_to_s3(file.file, bucket_name, object_name)
    if is_file_uploaded:
        return "File was successfully uplaoded"
    return "File could not be uplaoded"



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


@app.get("/validate/hash")
async def validate_hashes():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/report/models")
async def get_model_subgraph():
    raise HTTPException(status_code=501, detail="Not implemented")

import io
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from pandas import DataFrame
from sqlalchemy import Engine

from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.graph import ResolutionGraph
from matchbox.common.sources import Source, SourceAddress, SourceColumn
from matchbox.server import app
from matchbox.server.api import s3_to_recordbatch, table_to_s3
from matchbox.server.base import MatchboxDBAdapter

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

client = TestClient(app)


def test_healthcheck(self):
    """Test the healthcheck endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


@patch("matchbox.server.base.BackendManager.get_backend")
def test_count_all_backend_items(get_backend):
    """Test the unparameterised entity counting endpoint."""
    entity_counts = {
        "datasets": 1,
        "models": 2,
        "data": 3,
        "clusters": 4,
        "creates": 5,
        "merges": 6,
        "proposes": 7,
    }
    mock_backend = Mock()
    for e, c in entity_counts.items():
        mock_e = Mock()
        mock_e.count = Mock(return_value=c)
        setattr(mock_backend, e, mock_e)
    get_backend.return_value = mock_backend

    response = client.get("/testing/count")
    assert response.status_code == 200
    assert response.json() == {"entities": entity_counts}


@patch("matchbox.server.base.BackendManager.get_backend")
def test_count_backend_item(get_backend: MatchboxDBAdapter):
    """Test the parameterised entity counting endpoint."""
    mock_backend = Mock()
    mock_backend.models.count = Mock(return_value=20)
    get_backend.return_value = mock_backend

    response = client.get("/testing/count", params={"entity": "models"})
    assert response.status_code == 200
    assert response.json() == {"entities": {"models": 20}}


# def test_clear_backend():
#     response = client.post("/testing/clear")
#     assert response.status_code == 200


def test_add_source(s3: S3Client, warehouse_engine: Engine):
    """Test the source addition endpoint."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.index = Mock(return_value=None)

    # Build call
    hashes_table = pa.Table.from_pydict(
        {
            "source_pk": [
                [["short"], ["medium_id"]],
                [["very_long_identifier"], ["id"]],
            ],
            "hash": [[b"hash1", b"hash2"], [b"hash3", b"hash4"]],
        }
    )
    hashes_buffer = io.BytesIO()
    pa.parquet.write_table(hashes_table, hashes_buffer)
    hashes_buffer.seek(0)

    source = Source(
        address=SourceAddress.compose(full_name="test.source", engine=warehouse_engine),
        db_pk="pk",
        columns=[SourceColumn(name="company_name", alias="name")],
    )

    with patch(
        "matchbox.server.base.BackendManager.get_backend", return_value=mock_backend
    ):
        # Make request
        response = client.post(
            "/sources",
            data={
                "source": source.model_dump_json(),
            },
            files={
                # "source": (None, source.model_dump(), "application/json"),
                "data": ("hashes.parquet", hashes_buffer, "application/x-parquet"),
            },
        )

        # Validate response
        assert response.status_code == 200, response.json()
        assert response.json() == {"status": "ready"}
        mock_backend.index.assert_called_once()


# def test_list_sources():
#     response = client.get("/sources")
#     assert response.status_code == 200

# def test_get_source():
#     response = client.get("/sources/test_source")
#     assert response.status_code == 200

# def test_list_models():
#     response = client.get("/models")
#     assert response.status_code == 200

# def test_get_resolution():
#     response = client.get("/models/test_resolution")
#     assert response.status_code == 200

# def test_add_model():
#     response = client.post("/models")
#     assert response.status_code == 200

# def test_delete_model():
#     response = client.delete("/models/test_model")
#     assert response.status_code == 200

# def test_get_results():
#     response = client.get("/models/test_model/results")
#     assert response.status_code == 200

# def test_set_results():
#     response = client.post("/models/test_model/results")
#     assert response.status_code == 200

# def test_get_truth():
#     response = client.get("/models/test_model/truth")
#     assert response.status_code == 200

# def test_set_truth():
#     response = client.post("/models/test_model/truth")
#     assert response.status_code == 200

# def test_get_ancestors():
#     response = client.get("/models/test_model/ancestors")
#     assert response.status_code == 200

# def test_get_ancestors_cache():
#     response = client.get("/models/test_model/ancestors_cache")
#     assert response.status_code == 200

# def test_set_ancestors_cache():
#     response = client.post("/models/test_model/ancestors_cache")
#     assert response.status_code == 200

# def test_query():
#     response = client.get("/query")
#     assert response.status_code == 200

# def test_validate_ids():
#     response = client.get("/validate/id")
#     assert response.status_code == 200


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_resolution_graph(
    get_backend: MatchboxDBAdapter, resolution_graph: ResolutionGraph
):
    """Test the resolution graph report endpoint."""
    mock_backend = Mock()
    mock_backend.get_resolution_graph = Mock(return_value=resolution_graph)
    get_backend.return_value = mock_backend

    response = client.get("/report/resolutions")
    assert response.status_code == 200
    assert ResolutionGraph.model_validate(response.json())


@pytest.mark.asyncio
async def test_file_to_s3(s3: S3Client, all_companies: DataFrame):
    """Test that a file can be uploaded to S3."""
    # Create a mock bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Test 1: Upload a parquet file
    # Create a mock UploadFile
    all_companies["id"] = all_companies["id"].astype(str)
    table = pa.Table.from_pandas(all_companies)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    file_content = sink.getvalue().to_pybytes()

    parquet_file = UploadFile(filename="test.parquet", file=io.BytesIO(file_content))

    # Call the function
    upload_id = await table_to_s3(
        client=s3, bucket="test-bucket", file=parquet_file, expected_schema=table.schema
    )
    # Validate response
    assert UUID(upload_id, version=4)

    response_table = pa.Table.from_batches(
        [
            batch
            async for batch in s3_to_recordbatch(
                client=s3, bucket="test-bucket", key=f"{upload_id}.parquet"
            )
        ]
    )

    assert response_table.equals(table)

    # Test 2: Upload a non-parquet file
    text_file = UploadFile(filename="test.txt", file=io.BytesIO(b"test"))

    with pytest.raises(MatchboxServerFileError):
        await table_to_s3(
            client=s3,
            bucket="test-bucket",
            file=text_file,
            expected_schema=table.schema,
        )

    # Test 3: Upload a parquet file with a different schema
    corrupted_schema = table.schema.remove(0)
    with pytest.raises(MatchboxServerFileError):
        await table_to_s3(
            client=s3,
            bucket="test-bucket",
            file=parquet_file,
            expected_schema=corrupted_schema,
        )

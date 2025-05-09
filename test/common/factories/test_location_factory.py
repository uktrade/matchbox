"""Tests for the location factory and associated objects."""

from typing import Literal
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest
from faker import Faker
from sqlalchemy import Engine

from matchbox.common.dtos import DataTypes
from matchbox.common.factories.locations import (
    RelationalDBConfig,
    RelationalDBConfigOptions,
    location_factory,
)
from matchbox.common.sources import RelationalDBLocation, SourceField


class TestRelationalDBConfig:
    """Tests for the RelationalDBConfig class."""

    def test_creation_and_validation(self, sqlite_warehouse: Engine) -> None:
        """Test creating a RelationalDBConfig and validating its attributes."""
        uri = str(sqlite_warehouse.url)

        config = RelationalDBConfig(uri=uri)

        assert config.location_type == "rdbms"
        assert str(config.uri) == uri

        # Test default options
        assert config.location_options.table_strategy == "single"
        assert config.location_options.table_mapping is None

    def test_to_location(self, sqlite_warehouse: Engine) -> None:
        """Test converting a RelationalDBConfig to a RelationalDBLocation."""
        uri = str(sqlite_warehouse.url)

        config = RelationalDBConfig(uri=uri)

        location = config.to_location()

        assert isinstance(location, RelationalDBLocation)
        assert str(location.uri) == uri

    @pytest.mark.parametrize(
        ("table_strategy", "table_mapping", "has_joins"),
        [
            pytest.param("single", None, False, id="single_table_strategy"),
            pytest.param("spread", None, True, id="spread_table_strategy"),
            pytest.param(
                "single",
                {
                    "table1": (
                        SourceField(
                            name="jobcode",
                            type=DataTypes.STRING,
                        ),
                        SourceField(
                            name="name",
                            type=DataTypes.STRING,
                        ),
                    ),
                    "table2": (
                        SourceField(
                            name="age",
                            type=DataTypes.INT64,
                        ),
                    ),
                },
                True,
                id="custom_table_mapping",
            ),
        ],
    )
    def test_to_et_and_location_writer(
        self,
        sqlite_warehouse: Engine,
        table_strategy: Literal["single", "spread"],
        table_mapping: dict[str, list[SourceField]] | None,
        has_joins: bool,
    ) -> None:
        """Test to_et_and_location_writer with different table strategies."""
        # Create a Faker instance with fixed seed
        faker = Faker()
        Faker.seed(12345)

        # Define feature configs with faker generators
        field_configs = (
            SourceField(
                name="jobcode",
                type=DataTypes.STRING,
            ),
            SourceField(
                name="name",
                type=DataTypes.STRING,
            ),
            SourceField(
                name="age",
                type=DataTypes.INT64,
            ),
        )

        uri = str(sqlite_warehouse.url)

        config = RelationalDBConfig(
            uri=uri,
            location_options=RelationalDBConfigOptions(
                table_strategy=table_strategy,
                table_mapping=table_mapping,
            ),
        )

        sql, writer = config.to_et_and_location_writer(
            identifier=SourceField(name="pk", type=DataTypes.STRING),
            fields=field_configs,
            generator=faker,
        )

        # Check that the SQL is valid and references all fields
        assert "SELECT" in sql
        if has_joins:
            assert "LEFT JOIN" in sql

        for field in field_configs:
            assert f'"{field.name}"' in sql

        # If we have a custom mapping, check that the tables are in the SQL
        if table_mapping:
            for table in table_mapping:
                assert f'"{table}"' in sql

        # Verify the writer function exists
        assert callable(writer)

    def test_custom_mapping_validation(self, sqlite_warehouse: Engine) -> None:
        """Test validation of custom table mapping."""
        # Create a Faker instance with fixed seed
        faker = Faker()
        Faker.seed(12345)

        # Define field configs with faker generators
        field_configs = (
            SourceField(
                name="jobcode",
                type=DataTypes.STRING,
            ),
            SourceField(
                name="name",
                type=DataTypes.STRING,
            ),
            SourceField(
                name="age",
                type=DataTypes.INT64,
            ),
        )

        uri = str(sqlite_warehouse.url)

        # Incomplete mapping (missing fields)
        incomplete_mapping = {
            "table1": [field_configs[0]],  # Only id
        }

        config = RelationalDBConfig(
            uri=uri,
            location_options=RelationalDBConfigOptions(
                table_mapping=incomplete_mapping,
            ),
        )

        # This should raise a ValueError due to incomplete mapping
        with pytest.raises(ValueError):
            config.to_et_and_location_writer(
                identifier=SourceField(name="pk", type=DataTypes.STRING),
                fields=field_configs,
                generator=faker,
            )

    @pytest.mark.parametrize(
        ("table_strategy", "table_mapping", "expected_calls"),
        [
            pytest.param("single", None, 1, id="single_table_writer"),
            pytest.param(
                "spread", None, 4, id="spread_table_writer"
            ),  # 3 fields + 1 main table
            pytest.param(
                "single",
                {
                    "table1": (
                        SourceField(
                            name="jobcode",
                            type=DataTypes.STRING,
                        ),
                        SourceField(
                            name="name",
                            type=DataTypes.STRING,
                        ),
                    ),
                    "table2": (
                        SourceField(
                            name="age",
                            type=DataTypes.INT64,
                        ),
                    ),
                },
                2,
                id="custom_mapping_writer",
            ),
        ],
    )
    @patch("pandas.DataFrame.to_sql")
    def test_location_writer(
        self,
        mock_to_sql: MagicMock,
        sqlite_warehouse: Engine,
        table_strategy: Literal["single", "spread"],
        table_mapping: dict[str, list[SourceField]] | None,
        expected_calls: int,
    ) -> None:
        """Test the writer function with different table strategies."""
        # Create a Faker instance with fixed seed
        faker = Faker()
        Faker.seed(12345)

        # Define field configs with faker generators
        field_configs = (
            SourceField(
                name="jobcode",
                type=DataTypes.STRING,
            ),
            SourceField(
                name="name",
                type=DataTypes.STRING,
            ),
            SourceField(
                name="age",
                type=DataTypes.INT64,
            ),
        )

        # Create sample data
        sample_data = pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "pk": ["1", "2", "3"],
                    "jobcode": ["ID1", "ID2", "ID3"],
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )
        )

        # Create mock location
        mock_location = MagicMock(spec=RelationalDBLocation)
        mock_location.uri = str(sqlite_warehouse.url)

        config = RelationalDBConfig(
            uri=str(sqlite_warehouse.url),
            location_options=RelationalDBConfigOptions(
                table_strategy=table_strategy,
                table_mapping=table_mapping,
            ),
        )

        _, writer = config.to_et_and_location_writer(
            identifier=SourceField(name="pk", type=DataTypes.STRING),
            fields=field_configs,
            generator=faker,
        )

        # Call the writer function
        writer(sample_data, mock_location, sqlite_warehouse)

        # Check that to_sql was called the expected number of times
        assert mock_to_sql.call_count == expected_calls

        # Check arguments of the first call
        _, kwargs = mock_to_sql.call_args_list[0]
        assert kwargs["con"] == sqlite_warehouse
        assert kwargs["index"] is False
        assert kwargs["if_exists"] == "replace"


class TestLocationFactory:
    """Tests for the location_factory function."""

    def test_default_rdbms_location(self) -> None:
        """Test creating a default RDBMS location."""
        location_config = location_factory(location_type="rdbms")

        assert isinstance(location_config, RelationalDBConfig)
        assert location_config.location_type == "rdbms"
        assert str(location_config.uri) == "sqlite:///:memory:"

        # Check default options
        assert location_config.location_options.table_strategy == "single"
        assert location_config.location_options.table_mapping is None

    def test_custom_rdbms_location(self, sqlite_warehouse: Engine) -> None:
        """Test creating a custom RDBMS location."""
        uri = str(sqlite_warehouse.url)
        location_options = RelationalDBConfigOptions(table_strategy="spread")

        location_config = location_factory(
            location_type="rdbms",
            location_options=location_options,
            uri=uri,
        )

        assert isinstance(location_config, RelationalDBConfig)
        assert location_config.location_type == "rdbms"
        assert str(location_config.uri) == uri
        assert location_config.location_options.table_strategy == "spread"

    @pytest.mark.parametrize(
        ("location_type", "options", "exception_expected"),
        [
            pytest.param("rdbms", None, False, id="valjobcodelocation_type"),
            pytest.param(
                "unsupported_type", None, True, id="unsupported_location_type"
            ),
            pytest.param(
                "rdbms",
                {"table_strategy": "invalid"},
                True,
                id="invaljobcodeoptions_type",
            ),
        ],
    )
    def test_location_factory_validation(
        self, location_type: str, options: dict | None, exception_expected: bool
    ) -> None:
        """Test validation in location_factory."""
        if exception_expected:
            with pytest.raises(ValueError):
                location_factory(
                    location_type=location_type,
                    location_options=options,
                )
        else:
            config = location_factory(
                location_type=location_type,
                location_options=options,
            )
            assert isinstance(config, RelationalDBConfig)

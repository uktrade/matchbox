# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Matchbox** is a record matching and data linking orchestration framework developed by the UK Department for Business and Trade. It enables collaborative, measurable, and iterative entity matching without organizations sharing sensitive raw data.

## Technology Stack

- **Language**: Python 3.11-3.13
- **Data Processing**: Polars, DuckDB, PyArrow
- **Web Framework**: FastAPI (server), Streamlit (evaluation UI)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Storage**: MinIO (S3-compatible object storage)
- **Record Linking**: Splink library integration
- **Package Management**: uv
- **Task Runner**: just
- **Code Quality**: Ruff (linting/formatting)
- **Testing**: pytest
- **Documentation**: MkDocs with Material theme

## Essential Commands

### Development Workflow
```bash
# Initial setup
uv sync --all-extras
just build                    # Build containers and start services

# Testing
just test                     # Run full test suite
just test local              # Tests without Docker dependencies
just test docker             # Docker-dependent tests only

# Code quality
just format                   # Ruff formatting and linting

# Database migrations
just migration-generate "description"  # Create new migration
just migration-apply         # Apply pending migrations
just migration-check         # Verify migration status

# Development servers
just eval                     # Start Streamlit evaluation UI
just docs                     # Start MkDocs development server

# Security
just scan                     # TruffleHog secret scanning
```

### Testing Individual Components
```bash
# Run specific test files
pytest test/client/test_models.py
pytest test/server/test_api.py -v
pytest -k "test_linker" --no-header
```

## Architecture Overview

The project uses a **client-server architecture** with three main packages:

### 1. Client (`/src/matchbox/client/`)
- **Purpose**: Data analysis, model creation, and result querying
- **Key Components**:
  - `models/`: Linkers (deterministic, Splink-based) and dedupers
  - `helpers/`: Data selection, querying, matching utilities
  - `eval/`: Streamlit evaluation interface
  - `dags.py`: DAG management for processing pipelines
  - `results.py`: Result retrieval and processing

### 2. Server (`/src/matchbox/server/`)
- **Purpose**: REST API backend with database management
- **Key Components**:
  - `api/`: FastAPI application with routers
  - `postgresql/`: Database adapter, ORM models, Alembic migrations
  - Multi-stage Docker setup (dev/prod)

### 3. Common (`/src/matchbox/common/`)
- **Purpose**: Shared utilities and data structures
- **Key Components**:
  - `dtos.py`: Pydantic data transfer objects
  - `sources.py`: Source configurations and field definitions
  - `graph.py`: Resolution graph management
  - `factories/`: Data generation for testing

## Key Concepts

- **Sources**: External data configurations with field mappings
- **Models**: Linking/deduping methodologies (deterministic or probabilistic via Splink)
- **Resolutions**: Combinations of model outputs, sources, and human judgments
- **Resolution Graph**: Hierarchical structure tracking model dependencies
- **Clusters**: Groups of matched entities with probability scores
- **Index/Results Schema**: Standardized Arrow/Polars data formats

## Data Storage Architecture

- **PostgreSQL**: Metadata, configurations, and small results
- **MinIO/S3**: Large datasets stored as Arrow/Parquet files
- **DuckDB**: In-memory analytical processing

## Environment Setup

1. Copy `/environments/development.env` to `.env`
2. Key environment files:
   - `/environments/server.env` - Server configuration
   - `/environments/sample_client.env` - Client configuration template

## Testing Strategy

- **Unit tests**: Component-level testing
- **Integration tests**: Cross-component functionality  
- **End-to-end tests**: Full pipeline testing (`/test/e2e/`)
- **Docker markers**: Tests requiring containerized services use `@pytest.mark.docker`
- **Test fixtures**: Reusable data in `/test/fixtures/`

There is a house style for parameterising unit tests.

```python
@pytest.mark.parametrize(
    ["foo", "bar"],
    [
        pytest.param(True, 12, id="test_thing"),
        pytest.param(False, 16, id="test_other_thing"),
    ],
)
def test_something(foo: bool, bar: int):
    """Tests that something does something."""
```

### Factory System and Test Scenarios

The project uses a sophisticated **factory system** for generating realistic test data and complex integration scenarios. This system is essential for testing entity resolution workflows end-to-end.

#### Core Factory Components (`/src/matchbox/common/factories/`)

1. **Entities** (`entities.py`):
   - `SourceEntity`: Represents a true entity that can appear across multiple sources
   - `ClusterEntity`: Groups of matched records with shared identifiers
   - `VariationRule`: Rules for generating data variations (suffixes, prefixes, replacements)
   - `FeatureConfig`: Configures data generation using Faker generators

2. **Sources** (`sources.py`):
   - `SourceTestkit`: Complete generated source with data, hashes, and entity tracking
   - `LinkedSourcesTestkit`: Container for multiple related sources sharing true entities
   - `source_factory()`: Generates individual sources with configurable features
   - `linked_sources_factory()`: Generates linked sources (e.g., CRN, DUNS, CDMS datasets)

3. **Models** (`models.py`):
   - `ModelTestkit`: Generated models with probabilities and entity tracking
   - `query_to_model_factory()`: Creates linking/deduping models from query results
   - Component validation and probability generation

4. **DAGs** (`dags.py`):
   - `TestkitDAG`: Container managing sources, models, and their dependencies
   - Tracks resolution graph relationships between sources and models
   - Validates dependencies and maintains entity lineage

#### Scenario System (`/test/fixtures/db.py`)

The scenario system provides **canned integration test scenarios** with realistic data pipelines:

- **Scenario Types**:
  - `bare`: Basic linked sources (CRN, DUNS, CDMS) with no models
  - `index`: Sources indexed into Matchbox backend
  - `dedupe`: Deduplication models added to indexed sources
  - `link`: Cross-source linking models between deduplicated sources
  - `probabilistic_dedupe`: Probabilistic deduplication with thresholds
  - `alt_dedupe`: Alternative deduplication strategies for comparison
  - `convergent`: Multiple sources with near-identical indexing

- **Usage Pattern**:
  ```python
  with self.scenario(self.backend, "dedupe") as dag:
      # Access generated sources
      sources = dag.sources  # Dict of SourceTestkit objects
      models = dag.models    # Dict of ModelTestkit objects
      
      # Get real model names for testing
      model_name = list(dag.models.keys())[0]
  ```

- **Backend Integration**: 
  - Scenarios automatically populate PostgreSQL backend with realistic data
  - Sources are written to warehouse databases (SQLite/PostgreSQL)
  - Models and results are inserted into Matchbox backend
  - Caching system avoids regenerating expensive scenarios

#### Key Factory Concepts

- **Entity Tracking**: True entities are maintained across the entire pipeline
- **Data Lineage**: Track which keys/records belong to which entities
- **Realistic Data**: Uses Faker to generate company names, CRNs, addresses, etc.
- **Variations**: Systematic data variations (e.g., "Company Ltd" vs "Company Limited")
- **Cross-Source Linking**: Entities can appear in multiple sources with different keys
- **Probability Generation**: Models generate realistic probability scores for matches

This factory system enables **comprehensive integration testing** where you can:
- Generate complex multi-source datasets
- Test deduplication and linking algorithms
- Validate entity resolution pipelines end-to-end
- Compare model results against ground truth
- Test evaluation workflows with realistic data

The factory system is the foundation for all scenario-based testing and provides the infrastructure for testing components like the Textual evaluation UI with realistic data pipelines.

## Development Notes

- **Pre-commit hooks**: Automatic formatting, linting, docs build, and migration checks
- **No raw data storage**: Only indexes and metadata are stored
- **Splink integration**: Advanced probabilistic record linking capabilities
- **Security-first design**: Permission boundaries preserved, secrets scanning enabled
- **API-first approach**: All client operations go through the FastAPI server
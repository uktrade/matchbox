# Matchbox Project Overview

This document provides a comprehensive overview of the Matchbox project, its structure, and development conventions.

## Project Purpose

Matchbox is a Python-based open-source tool designed for record matching and identity resolution. It provides a framework for orchestrating and comparing data linking and deduplication methodologies. The project consists of a client and a server component, allowing for flexible deployment and usage scenarios.

## Technologies and Architecture

-   **Backend:** Python (3.11+)
-   **Data Processing:** Polars, DuckDB, PyArrow
-   **Web Framework:** FastAPI (server), Streamlit (evaluation UI)
-   **Database:** PostgreSQL with SQLAlchemy ORM
-   **Storage:** MinIO (S3-compatible object storage)
-   **Record Linking:** Splink library integration
-   **Package Management:** uv
-   **Task Runner:** just
-   **Code Quality:** Ruff (linting/formatting)
-   **Testing:** pytest
-   **Documentation:** MkDocs with the Material theme
-   **CI/CD:** GitHub Actions are used for running tests, linting, and deploying packages.

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

## Building and Running

The project uses `just` as a command runner to simplify common development tasks.

-   **Install dependencies:**
    ```bash
    uv sync --all-extras
    ```
-   **Run all tests:**
    ```bash
    just test
    ```
-   **Run specific tests:**
    To run a specific test file, use `uv run pytest`:
    ```bash
    uv run pytest test/path/to/your/test_file.py
    ```
-   **Run the server locally:**
    ```bash
    just build -d
    ```
-   **Format and lint the code:**
    ```bash
    just format
    ```
-   **Build and serve the documentation locally:**
    ```bash
    just docs
    ```

For a full list of available commands, run `just -l`.

## Development Conventions

-   **Code Style:** The project follows the Google Python Style Guide. Code is formatted and linted with `ruff`.
-   **Testing:** All code should be unit-tested using `pytest`. Tests are located in the `test/` directory.
-   **Pre-commit Hooks:** The project uses `pre-commit` to enforce code quality standards before committing. To install the hooks, run:
    ```bash
    pre-commit install
    ```
-   **Database Migrations:** Database migrations are managed with Alembic. To generate a new migration, run:
    ```bash
    just migration-generate "Your migration message"
    ```
-   **Dependency Management:** Dependencies are managed with `uv` and are listed in the `pyproject.toml` file.
-   **Documentation:** The project documentation is written in Markdown and generated using `MkDocs`. The documentation source is in the `docs/` directory.

## Testing Strategy

- **Unit tests**: Component-level testing
- **Integration tests**: Cross-component functionality
- **End-to-end tests**: Full pipeline testing (`/test/e2e/`)
- **Docker markers**: Tests requiring containerized services use `@pytest.mark.docker`
- **Test fixtures**: Reusable data in `/test/fixtures/`

### Factory System and Test Scenarios

The project uses a sophisticated **factory system** for generating realistic test data and complex integration scenarios. This system is essential for testing entity resolution workflows end-to-end. The factory system is the foundation for all scenario-based testing and provides the infrastructure for testing components like the Textual evaluation UI with realistic data pipelines.
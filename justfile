# Build and run all containers
build *DOCKER_ARGS:
    uv sync --extra server
    MB_VERSION=$(uv run --frozen python -m setuptools_scm) \
    docker compose --env-file=environments/containers.env up --build {{DOCKER_ARGS}}

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Run a local documentation development server
docs:
    uv run mkdocs serve

# Reformat and lint
format:
    uv run ruff format .
    uv run ruff check . --fix
    uvx uv-sort pyproject.toml

# Scan for secrets
scan:
    bash -c "docker run -v "$(pwd):/repo" -i \
        --rm trufflesecurity/trufflehog:latest git \
        file:///repo  --since-commit HEAD --fail"

# Run Python tests (usage: just test [local|docker])
test ENV="":
    #!/usr/bin/env bash
    if [[ "{{ENV}}" == "local" ]]; then
        uv run pytest -m "not docker"
    elif [[ "{{ENV}}" == "docker" ]]; then
        just build -d
        uv run pytest -m "docker"
    else
        just build -d
        uv run pytest
    fi

# Bring the database up to the latest migration script (the head)
migration-apply:
    uv run alembic --config "src/matchbox/server/postgresql/alembic.ini" upgrade head

# Check if migration-generate would produce a migration script without creating one
migration-check:
    uv run alembic --config "src/matchbox/server/postgresql/alembic.ini" check

# Autogenerate a new migration (keep your descriptive message brief as it is appended to the filename)
migration-generate descriptive-message:
    uv run alembic --config "src/matchbox/server/postgresql/alembic.ini" revision --autogenerate -m "{{descriptive-message}}"

# Reset the DB to the base state
migration-reset:
    uv run alembic --config "src/matchbox/server/postgresql/alembic.ini" downgrade base

# Run evaluation app with scenario data
eval scenario="" log="":
    uv run python test/scripts/eval.py {{scenario}} --log "{{log}}"

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Reformat and lint
format:
    uv run ruff format .
    uv run ruff check . --fix

# Scan for secrets
scan:
    bash -c "docker run -v "$(pwd):/repo" -i --rm trufflesecurity/trufflehog:latest git file:///repo"

# Run Python tests (usage: just test [local|docker])
test ENV="":
    #!/usr/bin/env bash
    if [[ "{{ENV}}" == "local" ]]; then
        uv run pytest -m "not docker"
    elif [[ "{{ENV}}" == "docker" ]]; then
        docker compose up --build -d --wait
        uv run pytest -m "docker"
    else
        docker compose up --build -d --wait
        uv run pytest
    fi

# Run a local documentation development server
docs:
    uv run mkdocs serve

# Autogenerate migrations
migrations-generate:
    docker compose down
    docker compose up -d matchbox-postgres
    uv run alembic revision --autogenerate
    echo "Please review alembic/versions to ensure expected as autogeneration of migrations can be incomplete"
    docker compose down

# Apply latest migration
migrations-apply:
    docker compose down
    docker compose up -d matchbox-postgres
    uv run alembic upgrade head
    docker compose down

# Drop all tables and re-create according to the current schema - this precludes any migrations
drop-recreate-tables:
    uv run python -c "from matchbox.client._handler import CLIENT; CLIENT.delete('/database', params={'certain': 'true'})"
    docker compose down

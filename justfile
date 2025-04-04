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

# Build and run all containers
build:
    docker compose up --build -d --wait

# Run Python tests (usage: just test [local|docker])
test ENV="":
    #!/usr/bin/env bash
    if [[ "{{ENV}}" == "local" ]]; then
        uv run pytest -m "not docker"
    elif [[ "{{ENV}}" == "docker" ]]; then
        just build
        uv run pytest -m "docker"
    else
        just build
        uv run pytest
    fi

# Run a local documentation development server
docs:
    uv run mkdocs serve

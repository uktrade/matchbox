# Unit testing
mod test 'test/justfile'
# PostgreSQL migration
mod migrate 'src/matchbox/server/postgresql/justfile'

# Build and run all containers
build *DOCKER_ARGS:
    uv sync --extra server
    MB_VERSION=$(uv run --frozen python -m setuptools_scm) \
    docker compose --env-file .env --env-file environments/containers.env up --build {{DOCKER_ARGS}}

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Run a local documentation development server
docs:
    uv run mkdocs serve --livereload

# Reformat and lint
format:
    uvx ruff@latest format .
    uvx ruff@latest check . --fix
    uvx uv-sort pyproject.toml

# Scan for secrets
scan:
    bash -c "docker run -v "$(pwd):/repo" -i \
        --rm trufflesecurity/trufflehog:latest git \
        file:///repo  --since-commit HEAD --fail"

# Run evaluation app with scenario data
eval scenario="" log="":
    uv run python test/scripts/eval.py {{scenario}} --log "{{log}}"

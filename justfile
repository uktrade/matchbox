# Make datasets table
matchbox:
    uv run python src/matchbox/admin.py --datasets datasets.toml

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Reformat and lint
format:
    uv run ruff format .
    uv run ruff check . --fix

# Run Python tests
test:
    docker compose up -d --wait
    uv run pytest

# Run development version of API
api:
    uv run fastapi dev src/matchbox/server/api.py
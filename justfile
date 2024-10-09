# Make datasets table
matchbox:
    uv run python src/matchbox/admin.py

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
    docker compose up db -d --wait
    uv run pytest

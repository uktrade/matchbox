FROM python:3.11 AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --all-extras --no-install-project --no-dev

ARG ENV_FILE
COPY ./uv.lock /code/uv.lock
COPY ./pyproject.toml /code/pyproject.toml
COPY ./src/matchbox /code/src/matchbox

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    # Determine package version dynamically from the git tag
    --mount=source=.git,target=/code/.git,type=bind \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MATCHBOX_DB=$(uv run python -m setuptools_scm) \
    uv sync --frozen --all-extras --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/code/.venv/bin:$PATH"

ENTRYPOINT ["ddtrace-run"]


# Local development stage: with locally-specified environment variables and hot reloading

FROM base AS dev

COPY ./environments/$ENV_FILE /code/.env
CMD ["fastapi", "dev", "--host", "0.0.0.0", "src/matchbox/server/api"]


# Production stage: no environment variables (they come from the infrastructure) and no hot reloading

FROM base AS prod

# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["fastapi", "run", "--host", "0.0.0.0", "src/matchbox/server/api"]

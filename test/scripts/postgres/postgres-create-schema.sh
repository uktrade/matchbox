#!/bin/bash
# Creates the Matchbox PostgreSQL schema and extensions on first container
# startup. Mounted into /docker-entrypoint-initdb.d/ by docker-compose so
# postgres runs it once when initialising a fresh data volume.
#
# These are operator responsibilities in production (see docs/server/install.md).
# This script automates them for local dev/CI only.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS ${MB__SERVER__POSTGRES__DB_SCHEMA:-mb};
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
EOSQL

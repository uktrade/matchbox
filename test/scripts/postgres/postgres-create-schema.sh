#!/bin/bash
# Creates the Matchbox PostgreSQL schema on first container startup.
# Mounted into /docker-entrypoint-initdb.d/ by docker-compose so postgres
# runs it once when initialising a fresh data volume.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS ${MB__SERVER__POSTGRES__DB_SCHEMA:-mb};
EOSQL

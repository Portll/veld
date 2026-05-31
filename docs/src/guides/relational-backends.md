# Relational backends

By default Veld is fully self-contained: the memory store runs on the local
embedded engine (RocksDB today, redb on the `v0.9` line) and the intent-log
projection writes to a local rusqlite slow store. **No external database is
required.**

For deployments that want memories — and the tabular *datasets* feature — to
live in an external SQL database, Veld can project through a pluggable
relational backend instead. This is opt-in and off by default: when
`VELD_RELATIONAL_BACKEND` is unset, behaviour is exactly the rusqlite-only
path, unchanged.

When a backend *is* configured, two things route through it:

- **The memories projection** — the intent-log apply path writes the
  `memories` table to the relational store (the W4 cutover).
- **Datasets** — the tabular-ingest and dataset MCP tools
  (`dataset_create`, `dataset_query`, `dataset_link`, …) create and query
  per-dataset tables plus the `__veld_dataset_catalog` and
  `__veld_dataset_row_links` bookkeeping tables. Tabular ingest returns
  `503 Service Unavailable` if no relational backend is configured.

## Selecting a backend

Set `VELD_RELATIONAL_BACKEND` to one of `sqlite`, `postgres`, `supabase`, or
`mssql`. Unset or empty keeps the default (rusqlite-only). Each backend reads
its own connection variable, and the networked backends must be compiled in
behind a feature flag — if you name a backend the binary wasn't built for,
startup fails fast with an actionable message rather than silently falling
back.

| `VELD_RELATIONAL_BACKEND` | Connection variable(s) | Build requirement |
|---|---|---|
| `sqlite`   | `VELD_RELATIONAL_SQLITE_PATH` | default build |
| `postgres` | `VELD_POSTGRES_URL` | `--features postgres` |
| `supabase` | `VELD_SUPABASE_PROJECT_REF` + `VELD_SUPABASE_DB_PASSWORD` | `--features postgres` |
| `mssql`    | `VELD_MSSQL_ADO_STRING` | `--features mssql` |

> **Secrets belong in the environment, never in the repo.** Connection
> strings and passwords below are secrets — export them in the service
> environment or a gitignored `.env`, and never commit them or pass them on a
> shared command line.

### SQLite (sqlx)

A file-backed SQLite database driven through `sqlx` — distinct from the
rusqlite slow store, and available in the default build with no extra feature
flag.

```bash
export VELD_RELATIONAL_BACKEND=sqlite
export VELD_RELATIONAL_SQLITE_PATH=/var/lib/veld/relational.db
```

### Postgres

```bash
# build:
cargo build --release --features postgres

# run:
export VELD_RELATIONAL_BACKEND=postgres
export VELD_POSTGRES_URL='postgres://veld:••••••@db.internal:5432/veld'
```

`VELD_POSTGRES_URL` is a standard `sqlx` Postgres connection string.

### Supabase

Supabase is Postgres underneath, so it also requires `--features postgres`.
Provide the **project ref** and the **database password** — the latter is the
Postgres role password from *Project Settings → Database*, **not** the
anon/service-role API JWTs.

```bash
cargo build --release --features postgres

export VELD_RELATIONAL_BACKEND=supabase
export VELD_SUPABASE_PROJECT_REF=abcdefghijklmnop
export VELD_SUPABASE_DB_PASSWORD='••••••'
```

Veld assembles the Postgres connection from the project ref and database
password.

### SQL Server (MSSQL)

Microsoft SQL Server via `tiberius`, behind `--features mssql`.
`VELD_MSSQL_ADO_STRING` is an ADO-style connection string (parsed by
tiberius' `Config::from_ado_string`).

```bash
cargo build --release --features mssql

export VELD_RELATIONAL_BACKEND=mssql
export VELD_MSSQL_ADO_STRING='Server=tcp:sql.internal,1433;Database=veld;User Id=veld;Password=••••••;TrustServerCertificate=true'
```

## Dialect handling

The DDL and SQL Veld emits is dialect-correct per backend — Postgres
(`BYTEA`, `BIGINT`, `ON CONFLICT DO NOTHING`, typed nulls), SQL Server
(`OBJECT_ID` create guards, `NVARCHAR`/`VARBINARY`, bounded primary-key
columns within the 900-byte index-key limit, `IF NOT EXISTS` idempotent
inserts, `SELECT TOP (n)`), and SQLite. You do not need to pre-create any
tables; Veld creates the `memories` table, dataset catalog, per-dataset
tables, and link table on first use.

## Status

The networked backends (Postgres, Supabase, SQL Server) are recent additions.
Their schema and queries are dialect-correct and covered by unit tests, but
they are not yet exercised against live database instances in CI — validate
against your own instance before relying on them in production. The embedded
SQLite paths are covered end-to-end by the test suite.

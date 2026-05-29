//! Dataset schema types — structural definition of a tabular dataset.
//!
//! These types describe what a dataset *looks like* (columns, types,
//! nullability, primary key) independently of where the rows are stored.
//! The [`DatasetSchema::to_create_table_sql_sqlite`] and
//! [`DatasetSchema::to_create_table_sql_postgres`] helpers produce DDL
//! strings; they never execute SQL themselves. A downstream agent wires the
//! schema to a real `RelationalStore`.

use serde::{Deserialize, Serialize};

/// Structural schema of a single dataset.
///
/// `name` is the user-facing dataset label (free-form). `columns` carries
/// the ordered list of column definitions. `primary_key` lists the column
/// names that together form the primary key — a single-column PK has one
/// entry; a composite PK has several.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetSchema {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Vec<String>,
}

/// Definition of a single column within a [`DatasetSchema`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub ty: ColumnType,
    pub nullable: bool,
}

/// Storage-engine-independent column type. The DDL rendering layer maps
/// each variant to the closest native type in SQLite or Postgres.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColumnType {
    Bool,
    I64,
    F64,
    Text,
    Bytes,
    Timestamp,
    Json,
}

impl ColumnType {
    /// SQL identifier rendering for SQLite.
    ///
    /// SQLite uses type *affinity* rather than strict types; booleans
    /// collapse to `INTEGER` (`0`/`1`), timestamps are stored as ISO 8601
    /// `TEXT`, and JSON uses `TEXT` (the bundled `JSON1` extension parses
    /// it on read — no schema-level annotation needed).
    fn sqlite_type(self) -> &'static str {
        match self {
            ColumnType::Bool => "INTEGER",
            ColumnType::I64 => "INTEGER",
            ColumnType::F64 => "REAL",
            ColumnType::Text => "TEXT",
            ColumnType::Bytes => "BLOB",
            ColumnType::Timestamp => "TEXT",
            ColumnType::Json => "TEXT",
        }
    }

    /// SQL identifier rendering for Postgres.
    fn postgres_type(self) -> &'static str {
        match self {
            ColumnType::Bool => "BOOLEAN",
            ColumnType::I64 => "BIGINT",
            ColumnType::F64 => "DOUBLE PRECISION",
            ColumnType::Text => "TEXT",
            ColumnType::Bytes => "BYTEA",
            ColumnType::Timestamp => "TIMESTAMPTZ",
            ColumnType::Json => "JSONB",
        }
    }
}

impl DatasetSchema {
    /// Render a `CREATE TABLE IF NOT EXISTS` statement targeting SQLite.
    ///
    /// `table_name` is used verbatim — callers are expected to have
    /// sanitised it via [`crate::datasets::store::sanitise_table_name`] or
    /// equivalent. Identifiers are quoted with double quotes.
    pub fn to_create_table_sql_sqlite(&self, table_name: &str) -> String {
        render_create_table(self, table_name, RenderDialect::Sqlite)
    }

    /// Render a `CREATE TABLE IF NOT EXISTS` statement targeting Postgres.
    pub fn to_create_table_sql_postgres(&self, table_name: &str) -> String {
        render_create_table(self, table_name, RenderDialect::Postgres)
    }
}

#[derive(Debug, Clone, Copy)]
enum RenderDialect {
    Sqlite,
    Postgres,
}

fn render_create_table(schema: &DatasetSchema, table_name: &str, dialect: RenderDialect) -> String {
    let mut out = String::with_capacity(128);
    out.push_str("CREATE TABLE IF NOT EXISTS \"");
    out.push_str(table_name);
    out.push_str("\" (\n");

    for (idx, col) in schema.columns.iter().enumerate() {
        out.push_str("  \"");
        out.push_str(&col.name);
        out.push_str("\" ");
        let ty = match dialect {
            RenderDialect::Sqlite => col.ty.sqlite_type(),
            RenderDialect::Postgres => col.ty.postgres_type(),
        };
        out.push_str(ty);
        if !col.nullable {
            out.push_str(" NOT NULL");
        }
        if idx + 1 < schema.columns.len() || !schema.primary_key.is_empty() {
            out.push(',');
        }
        out.push('\n');
    }

    if !schema.primary_key.is_empty() {
        out.push_str("  PRIMARY KEY (");
        for (idx, pk) in schema.primary_key.iter().enumerate() {
            if idx > 0 {
                out.push_str(", ");
            }
            out.push('"');
            out.push_str(pk);
            out.push('"');
        }
        out.push_str(")\n");
    }

    out.push_str(");");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> DatasetSchema {
        DatasetSchema {
            name: "events".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: ColumnType::I64,
                    nullable: false,
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "score".to_string(),
                    ty: ColumnType::F64,
                    nullable: true,
                },
                ColumnDef {
                    name: "active".to_string(),
                    ty: ColumnType::Bool,
                    nullable: false,
                },
                ColumnDef {
                    name: "payload".to_string(),
                    ty: ColumnType::Bytes,
                    nullable: true,
                },
                ColumnDef {
                    name: "at".to_string(),
                    ty: ColumnType::Timestamp,
                    nullable: false,
                },
                ColumnDef {
                    name: "meta".to_string(),
                    ty: ColumnType::Json,
                    nullable: true,
                },
            ],
            primary_key: vec!["id".to_string()],
        }
    }

    #[test]
    fn schema_round_trips_through_json() {
        let schema = sample_schema();
        let json = serde_json::to_string(&schema).expect("encode");
        let back: DatasetSchema = serde_json::from_str(&json).expect("decode");
        assert_eq!(schema, back);
    }

    #[test]
    fn create_table_sql_sqlite_renders_expected_types() {
        let sql = sample_schema().to_create_table_sql_sqlite("alice__dataset__events");
        let expected = "CREATE TABLE IF NOT EXISTS \"alice__dataset__events\" (\n  \"id\" INTEGER NOT NULL,\n  \"name\" TEXT NOT NULL,\n  \"score\" REAL,\n  \"active\" INTEGER NOT NULL,\n  \"payload\" BLOB,\n  \"at\" TEXT NOT NULL,\n  \"meta\" TEXT,\n  PRIMARY KEY (\"id\")\n);";
        assert_eq!(sql, expected);
    }

    #[test]
    fn create_table_sql_postgres_renders_expected_types() {
        let sql = sample_schema().to_create_table_sql_postgres("alice__dataset__events");
        let expected = "CREATE TABLE IF NOT EXISTS \"alice__dataset__events\" (\n  \"id\" BIGINT NOT NULL,\n  \"name\" TEXT NOT NULL,\n  \"score\" DOUBLE PRECISION,\n  \"active\" BOOLEAN NOT NULL,\n  \"payload\" BYTEA,\n  \"at\" TIMESTAMPTZ NOT NULL,\n  \"meta\" JSONB,\n  PRIMARY KEY (\"id\")\n);";
        assert_eq!(sql, expected);
    }
}

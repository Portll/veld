//! Tabular ingest — parse CSV or JSON-array-of-objects into a dataset
//! schema + rows for the W7 `RelationalDatasetStore`, instead of producing
//! memories.
//!
//! This is the data path behind the `tabular` flag on `/api/ingest`: when
//! set, structured input becomes one relational table (one row per record)
//! rather than one free-text memory. The parsers here are pure and unit
//! tested; the handler in `handlers::ingest` wires them to the store.
//!
//! ## Schema mapping
//!
//! Every source column becomes a nullable `TEXT` column — values are stored
//! as text and not type-inferred (a future refinement). A synthetic
//! [`ROW_ID_COLUMN`] (`__row`, `I64`) primary key is prepended so rows are
//! addressable (e.g. for `dataset_link`) even when the source has no key.
//! Non-string JSON cells are stringified so they fit the `TEXT` columns
//! uniformly across SQLite and Postgres (the latter rejects a bigint bound
//! into a text column).

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use serde_json::Value;

use crate::datasets::{ColumnDef, ColumnType, DatasetRow, DatasetSchema};
use crate::ingest::InputFormat;

/// Synthetic per-row primary-key column added to every tabular dataset.
pub const ROW_ID_COLUMN: &str = "__row";

/// Parse `content` (CSV or JSON array-of-objects) into a [`DatasetSchema`]
/// named `dataset_name` plus the rows to insert.
///
/// Errors when the format is not CSV/JSON, the input has no columns, or the
/// JSON is not an array of objects (a bare object is accepted as one row).
pub fn to_dataset(
    format: InputFormat,
    content: &str,
    dataset_name: &str,
) -> Result<(DatasetSchema, Vec<DatasetRow>)> {
    let (columns, raw_rows) = match format {
        InputFormat::Csv => parse_csv(content)?,
        InputFormat::Json => parse_json_tabular(content)?,
        other => bail!(
            "tabular ingest supports CSV or JSON, got {}",
            other.as_str()
        ),
    };
    if columns.is_empty() {
        bail!("tabular ingest found no columns in the input");
    }

    // Schema: synthetic row-id PK + one nullable TEXT column per source col.
    let mut schema_columns = Vec::with_capacity(columns.len() + 1);
    schema_columns.push(ColumnDef {
        name: ROW_ID_COLUMN.to_string(),
        ty: ColumnType::I64,
        nullable: false,
    });
    for col in &columns {
        schema_columns.push(ColumnDef {
            name: col.clone(),
            ty: ColumnType::Text,
            nullable: true,
        });
    }
    let schema = DatasetSchema {
        name: dataset_name.to_string(),
        columns: schema_columns,
        primary_key: vec![ROW_ID_COLUMN.to_string()],
    };

    let rows = raw_rows
        .into_iter()
        .enumerate()
        .map(|(i, cells)| {
            let mut values: HashMap<String, Value> = HashMap::with_capacity(columns.len() + 1);
            values.insert(ROW_ID_COLUMN.to_string(), Value::from(i as i64));
            for (col, cell) in columns.iter().zip(cells.into_iter()) {
                values.insert(col.clone(), text_cell(cell));
            }
            DatasetRow { values }
        })
        .collect();

    Ok((schema, rows))
}

/// Coerce a parsed cell into something a `TEXT` column accepts uniformly:
/// strings and nulls pass through; everything else (numbers, bools, nested
/// JSON) is rendered to its compact JSON string so neither SQLite nor
/// Postgres rejects the bind.
fn text_cell(cell: Value) -> Value {
    match cell {
        Value::String(_) | Value::Null => cell,
        other => Value::String(other.to_string()),
    }
}

/// Parse RFC-4180-style CSV: the first record is the header row, the rest
/// are data rows. Quote-aware — fields may be wrapped in double quotes and
/// contain commas, newlines, and escaped quotes (`""`). Each cell comes back
/// as a `Value::String`; short rows are padded with `Null`, long rows are
/// truncated to the header width.
fn parse_csv(content: &str) -> Result<(Vec<String>, Vec<Vec<Value>>)> {
    let mut records = parse_csv_records(content).into_iter();
    let headers = match records.next() {
        Some(h) if h.iter().any(|c| !c.is_empty()) => h,
        _ => bail!("CSV input has no header row"),
    };
    let ncols = headers.len();

    let mut rows = Vec::new();
    for rec in records {
        // Skip fully-blank lines (e.g. a trailing newline producing an
        // empty single-field record).
        if rec.len() <= 1 && rec.first().map(|c| c.is_empty()).unwrap_or(true) {
            continue;
        }
        let mut cells = Vec::with_capacity(ncols);
        for i in 0..ncols {
            match rec.get(i) {
                Some(s) => cells.push(Value::String(s.clone())),
                None => cells.push(Value::Null),
            }
        }
        rows.push(cells);
    }
    Ok((headers, rows))
}

/// Split raw CSV text into records of fields, honoring quoted fields with
/// embedded commas / newlines / escaped quotes.
fn parse_csv_records(content: &str) -> Vec<Vec<String>> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(ch);
            }
        } else {
            match ch {
                '"' => in_quotes = true,
                ',' => record.push(std::mem::take(&mut field)),
                '\r' => { /* swallow; the \n (or EOF) ends the record */ }
                '\n' => {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                }
                _ => field.push(ch),
            }
        }
    }
    // Flush a trailing field/record that wasn't newline-terminated.
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

/// Parse a JSON array of objects (or a single object) into ordered columns
/// (first-seen union of keys across all objects) and per-row cell vectors
/// aligned to those columns; absent keys become `Null`.
fn parse_json_tabular(content: &str) -> Result<(Vec<String>, Vec<Vec<Value>>)> {
    let value: Value =
        serde_json::from_str(content).map_err(|e| anyhow::anyhow!("invalid JSON: {e}"))?;
    let array = match value {
        Value::Array(a) => a,
        obj @ Value::Object(_) => vec![obj],
        _ => bail!("tabular JSON must be an array of objects (or a single object)"),
    };

    let mut columns: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut objects = Vec::with_capacity(array.len());
    for el in array {
        let obj = match el {
            Value::Object(m) => m,
            _ => bail!("tabular JSON array elements must be objects"),
        };
        for key in obj.keys() {
            if seen.insert(key.clone()) {
                columns.push(key.clone());
            }
        }
        objects.push(obj);
    }

    let rows = objects
        .into_iter()
        .map(|obj| {
            columns
                .iter()
                .map(|c| obj.get(c).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    Ok((columns, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_simple_headers_and_rows() {
        let (schema, rows) =
            to_dataset(InputFormat::Csv, "name,age\nalice,30\nbob,25\n", "people").unwrap();
        // __row PK + 2 source columns.
        assert_eq!(schema.columns.len(), 3);
        assert_eq!(schema.columns[0].name, ROW_ID_COLUMN);
        assert_eq!(schema.primary_key, vec![ROW_ID_COLUMN.to_string()]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values.get("name").unwrap(), &Value::String("alice".into()));
        assert_eq!(rows[0].values.get(ROW_ID_COLUMN).unwrap(), &Value::from(0i64));
        assert_eq!(rows[1].values.get("age").unwrap(), &Value::String("25".into()));
        assert_eq!(rows[1].values.get(ROW_ID_COLUMN).unwrap(), &Value::from(1i64));
    }

    #[test]
    fn csv_quoted_fields_with_commas_and_newlines() {
        let csv = "a,b\n\"x,y\",\"line1\nline2\"\n";
        let (_schema, rows) = to_dataset(InputFormat::Csv, csv, "d").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values.get("a").unwrap(), &Value::String("x,y".into()));
        assert_eq!(rows[0].values.get("b").unwrap(), &Value::String("line1\nline2".into()));
    }

    #[test]
    fn csv_escaped_quotes_and_short_rows() {
        let csv = "a,b\n\"he said \"\"hi\"\"\",only_a\nsolo";
        let (_s, rows) = to_dataset(InputFormat::Csv, csv, "d").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values.get("a").unwrap(), &Value::String("he said \"hi\"".into()));
        // Short row: missing 'b' padded to Null.
        assert_eq!(rows[1].values.get("a").unwrap(), &Value::String("solo".into()));
        assert_eq!(rows[1].values.get("b").unwrap(), &Value::Null);
    }

    #[test]
    fn json_array_of_objects_unions_keys() {
        let json = r#"[{"id":1,"name":"a"},{"id":2,"city":"NYC"}]"#;
        let (schema, rows) = to_dataset(InputFormat::Json, json, "d").unwrap();
        // __row + union {id, name, city}.
        let col_names: Vec<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(col_names, vec![ROW_ID_COLUMN, "id", "name", "city"]);
        assert_eq!(rows.len(), 2);
        // Numbers stringified into TEXT cells.
        assert_eq!(rows[0].values.get("id").unwrap(), &Value::String("1".into()));
        assert_eq!(rows[0].values.get("name").unwrap(), &Value::String("a".into()));
        // Missing key → Null.
        assert_eq!(rows[0].values.get("city").unwrap(), &Value::Null);
        assert_eq!(rows[1].values.get("city").unwrap(), &Value::String("NYC".into()));
    }

    #[test]
    fn json_single_object_is_one_row() {
        let (_s, rows) = to_dataset(InputFormat::Json, r#"{"k":"v"}"#, "d").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values.get("k").unwrap(), &Value::String("v".into()));
    }

    #[test]
    fn json_non_object_elements_error() {
        assert!(to_dataset(InputFormat::Json, "[1,2,3]", "d").is_err());
    }

    #[test]
    fn non_tabular_format_errors() {
        assert!(to_dataset(InputFormat::Markdown, "# hi", "d").is_err());
    }
}

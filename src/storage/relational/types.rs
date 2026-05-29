//! Value and metadata types shared by all `RelationalStore` backends.
//!
//! These are deliberately backend-agnostic. The SQLite, Postgres, and Supabase
//! adapters all funnel their native column representations through
//! [`ColumnValue`] and accept [`Param`] for bound parameters.

use std::fmt;

/// Identifier for the concrete backend behind a `RelationalStore`.
///
/// Higher-level code uses this to make capability decisions (e.g. whether to
/// rely on SQLite-only `INSERT OR REPLACE`, or Postgres-specific `RETURNING`)
/// without resorting to runtime SQL probing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalBackend {
    Sqlite,
    Postgres,
    Supabase,
    Mssql,
}

impl fmt::Display for RelationalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            RelationalBackend::Sqlite => "sqlite",
            RelationalBackend::Postgres => "postgres",
            RelationalBackend::Supabase => "supabase",
            RelationalBackend::Mssql => "mssql",
        };
        f.write_str(name)
    }
}

/// Borrowed parameter value used to bind a placeholder in a SQL statement.
///
/// Backends translate this enum into their native bind-call sequence.
#[derive(Debug, Clone)]
pub enum Param<'a> {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Text(&'a str),
    Bytes(&'a [u8]),
    Json(&'a serde_json::Value),
}

/// Backend-neutral column metadata.
///
/// `sql_type` is the backend-reported type string; consumers may inspect it
/// but must not rely on it for portable logic. Use [`Row::get`] /
/// [`Row::get_by_name`] (with [`FromColumn`]) for typed access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMeta {
    pub name: String,
    pub sql_type: String,
}

/// Owned, backend-neutral representation of a single column value.
///
/// Rows hold these internally; callers see only the borrowed
/// [`ColumnValue`] view through [`FromColumn::from_column`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum OwnedColumnValue {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Text(String),
    Bytes(Vec<u8>),
    Json(serde_json::Value),
}

impl OwnedColumnValue {
    pub(crate) fn as_borrowed(&self) -> ColumnValue<'_> {
        match self {
            OwnedColumnValue::Null => ColumnValue::Null,
            OwnedColumnValue::Bool(b) => ColumnValue::Bool(*b),
            OwnedColumnValue::I64(i) => ColumnValue::I64(*i),
            OwnedColumnValue::F64(f) => ColumnValue::F64(*f),
            OwnedColumnValue::Text(s) => ColumnValue::Text(s.as_str()),
            OwnedColumnValue::Bytes(b) => ColumnValue::Bytes(b.as_slice()),
            OwnedColumnValue::Json(v) => ColumnValue::Json(v),
        }
    }
}

/// Borrowed view of a column value handed to [`FromColumn`].
///
/// All variants borrow from the [`Row`] that owns the underlying storage,
/// so converters can copy or zero-copy as appropriate.
#[derive(Debug, Clone, Copy)]
pub enum ColumnValue<'a> {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Text(&'a str),
    Bytes(&'a [u8]),
    Json(&'a serde_json::Value),
}

impl ColumnValue<'_> {
    /// Human-readable variant name, used in error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            ColumnValue::Null => "null",
            ColumnValue::Bool(_) => "bool",
            ColumnValue::I64(_) => "i64",
            ColumnValue::F64(_) => "f64",
            ColumnValue::Text(_) => "text",
            ColumnValue::Bytes(_) => "bytes",
            ColumnValue::Json(_) => "json",
        }
    }
}

/// Errors surfaced when decoding a column value into a Rust type.
#[derive(Debug, thiserror::Error)]
pub enum ColumnError {
    #[error("column index {index} out of range (row has {len} columns)")]
    IndexOutOfRange { index: usize, len: usize },

    #[error("column named {name:?} not found in row")]
    UnknownColumn { name: String },

    #[error("unexpected NULL value while decoding column to {target}")]
    UnexpectedNull { target: &'static str },

    #[error("cannot decode column of type {actual} into {target}")]
    TypeMismatch {
        actual: &'static str,
        target: &'static str,
    },

    #[error("malformed JSON column value: {0}")]
    InvalidJson(String),
}

/// Single row returned by [`RelationalStore::query`].
///
/// Rows own their values; metadata is shared across the row but separately
/// addressable via [`Row::columns`]. Use [`Row::get`] (by index) or
/// [`Row::get_by_name`] (by column name) for typed access.
#[derive(Debug, Clone)]
pub struct Row {
    columns: Vec<ColumnMeta>,
    values: Vec<OwnedColumnValue>,
}

impl Row {
    /// Construct a row from already-decoded column metadata and values.
    ///
    /// Backend adapters use this after converting native row data to the
    /// neutral [`OwnedColumnValue`] representation.
    pub(crate) fn new(columns: Vec<ColumnMeta>, values: Vec<OwnedColumnValue>) -> Self {
        debug_assert_eq!(columns.len(), values.len(), "row column/value arity mismatch");
        Self { columns, values }
    }

    /// Number of columns in this row.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when the row has zero columns.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrow the column metadata for this row.
    pub fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    /// Decode the column at `idx` into `T`.
    pub fn get<T: FromColumn>(&self, idx: usize) -> Result<T, ColumnError> {
        let value = self.values.get(idx).ok_or(ColumnError::IndexOutOfRange {
            index: idx,
            len: self.values.len(),
        })?;
        T::from_column(value.as_borrowed())
    }

    /// Decode the column with the given name into `T`.
    pub fn get_by_name<T: FromColumn>(&self, name: &str) -> Result<T, ColumnError> {
        let idx = self
            .columns
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| ColumnError::UnknownColumn {
                name: name.to_string(),
            })?;
        self.get(idx)
    }
}

/// Decoder from a backend-neutral [`ColumnValue`] into a Rust type.
///
/// Implementations are expected to be tolerant of common upcasts (e.g. an
/// `i64` column requested as `i64` succeeds; a Text column requested as
/// `String` succeeds; everything else returns [`ColumnError::TypeMismatch`]).
pub trait FromColumn: Sized {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError>;
}

impl FromColumn for bool {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError> {
        match value {
            ColumnValue::Bool(b) => Ok(b),
            // SQLite stores booleans as integers; accept the common 0/1 encoding.
            ColumnValue::I64(0) => Ok(false),
            ColumnValue::I64(1) => Ok(true),
            ColumnValue::Null => Err(ColumnError::UnexpectedNull { target: "bool" }),
            other => Err(ColumnError::TypeMismatch {
                actual: other.type_name(),
                target: "bool",
            }),
        }
    }
}

impl FromColumn for i64 {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError> {
        match value {
            ColumnValue::I64(i) => Ok(i),
            ColumnValue::Bool(b) => Ok(b as i64),
            ColumnValue::Null => Err(ColumnError::UnexpectedNull { target: "i64" }),
            other => Err(ColumnError::TypeMismatch {
                actual: other.type_name(),
                target: "i64",
            }),
        }
    }
}

impl FromColumn for f64 {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError> {
        match value {
            ColumnValue::F64(f) => Ok(f),
            ColumnValue::I64(i) => Ok(i as f64),
            ColumnValue::Null => Err(ColumnError::UnexpectedNull { target: "f64" }),
            other => Err(ColumnError::TypeMismatch {
                actual: other.type_name(),
                target: "f64",
            }),
        }
    }
}

impl FromColumn for String {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError> {
        match value {
            ColumnValue::Text(s) => Ok(s.to_string()),
            ColumnValue::Json(v) => Ok(v.to_string()),
            ColumnValue::Null => Err(ColumnError::UnexpectedNull { target: "String" }),
            other => Err(ColumnError::TypeMismatch {
                actual: other.type_name(),
                target: "String",
            }),
        }
    }
}

impl FromColumn for Vec<u8> {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError> {
        match value {
            ColumnValue::Bytes(b) => Ok(b.to_vec()),
            ColumnValue::Text(s) => Ok(s.as_bytes().to_vec()),
            ColumnValue::Null => Err(ColumnError::UnexpectedNull { target: "Vec<u8>" }),
            other => Err(ColumnError::TypeMismatch {
                actual: other.type_name(),
                target: "Vec<u8>",
            }),
        }
    }
}

impl FromColumn for serde_json::Value {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError> {
        match value {
            ColumnValue::Json(v) => Ok(v.clone()),
            // Text columns commonly hold JSON in SQLite; try to parse so the
            // round-trip from `Param::Json` survives storage.
            ColumnValue::Text(s) => {
                serde_json::from_str(s).map_err(|e| ColumnError::InvalidJson(e.to_string()))
            }
            ColumnValue::Bytes(b) => {
                serde_json::from_slice(b).map_err(|e| ColumnError::InvalidJson(e.to_string()))
            }
            ColumnValue::Null => Err(ColumnError::UnexpectedNull {
                target: "serde_json::Value",
            }),
            ColumnValue::Bool(b) => Ok(serde_json::Value::Bool(b)),
            ColumnValue::I64(i) => Ok(serde_json::Value::Number(i.into())),
            ColumnValue::F64(f) => serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .ok_or_else(|| {
                    ColumnError::InvalidJson(format!("non-finite f64 {f} not representable as JSON"))
                }),
        }
    }
}

impl<T: FromColumn> FromColumn for Option<T> {
    fn from_column(value: ColumnValue<'_>) -> Result<Self, ColumnError> {
        match value {
            ColumnValue::Null => Ok(None),
            other => T::from_column(other).map(Some),
        }
    }
}

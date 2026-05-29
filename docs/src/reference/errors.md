<!-- GENERATED FILE — do not edit by hand.
     Source: src/errors.rs
     Generator: docs/generators/src/bin/gen-errors.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-errors`) -->

# Error Reference

The error types defined in `src/errors.rs`. 1 error enum(s) discovered.

## `AppError`

21 variants.

| Variant | Message / Doc |
|---|---|
| `InvalidInput *(carries data)*` | — |
| `InvalidUserId *(carries data)*` | — |
| `InvalidMemoryId *(carries data)*` | — |
| `InvalidEmbeddings *(carries data)*` | — |
| `ContentTooLarge *(carries data)*` | — |
| `ResourceLimit *(carries data)*` | — |
| `AmbiguousMemoryId *(carries data)*` | — |
| `MemoryNotFound *(carries data)*` | — |
| `UserNotFound *(carries data)*` | — |
| `TodoNotFound *(carries data)*` | — |
| `ProjectNotFound *(carries data)*` | — |
| `ContextBlockNotFound *(carries data)*` | — |
| `MemoryAlreadyExists *(carries data)*` | — |
| `StorageError *(carries data)*` | — |
| `DatabaseError *(carries data)*` | — |
| `SerializationError *(carries data)*` | — |
| `ConcurrencyError *(carries data)*` | — |
| `LockPoisoned *(carries data)*` | — |
| `LockAcquisitionFailed *(carries data)*` | — |
| `ServiceUnavailable *(carries data)*` | — |
| `Internal *(carries data)*` | — |

---

*HTTP status mappings live in the `IntoResponse` impl on `AppError`. Inspect [src/errors.rs](https://github.com/Portll/veld/blob/main/src/errors.rs) for the full mapping.*

use pyo3::prelude::*;

#[pymodule]
fn veld(m: &Bound<'_, PyModule>) -> PyResult<()> {
    veld_core::python::register(m)
}

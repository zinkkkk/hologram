use pyo3::prelude::*;

pub mod py_rbf;
use py_rbf::PyRbf;

#[pymodule]
fn pyholo(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRbf>()?;
    Ok(())
}

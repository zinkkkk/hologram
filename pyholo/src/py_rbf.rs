use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use hologram::{
    kernels::{
        cubic_kernel, gaussian_kernel, inverse_multi_kernel, linear_kernel, multiquadric_kernel,
        thin_plate_spline_kernel,
    },
    rbf::Rbf,
    Interpolator,
};

#[derive(Clone)]
pub enum RbfKind {
    SS(Rbf<f64, f64>),
    SV(Rbf<f64, Vec<f64>>),
    VS(Rbf<Vec<f64>, f64>),
    VV(Rbf<Vec<f64>, Vec<f64>>),
}

#[pyclass(name = "Rbf")]
#[derive(Clone)]
pub struct PyRbf {
    inner: RbfKind,
}

#[pymethods]
impl PyRbf {
    #[new]
    pub fn new(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        kernel_name: Option<&str>,
        epsilon: Option<f64>,
    ) -> PyResult<Self> {
        let kernel = Some(match kernel_name.unwrap_or("gaussian") {
            "linear" => linear_kernel,
            "cubic" => cubic_kernel,
            "gaussian" => gaussian_kernel,
            "multiquadric" => multiquadric_kernel,
            "inverse_multiquadratic" => inverse_multi_kernel,
            "thin_plate_spline" => thin_plate_spline_kernel,
            _ => return Err(PyValueError::new_err("Unsupported kernel")),
        });

        let epsilon = epsilon.unwrap_or(1.0);

        macro_rules! try_rbf {
            ($x_ty:ty, $y_ty:ty, $variant:path) => {
                if let (Ok(x), Ok(y)) = (x.extract::<$x_ty>(), y.extract::<$y_ty>()) {
                    // Release GIL during the RBF construction / solving
                    return py
                        .allow_threads(move || {
                            Rbf::new(x, y, kernel, Some(epsilon))
                                .map($variant)
                                .map(|inner| PyRbf { inner })
                        })
                        .map_err(PyValueError::new_err);
                }
            };
        }

        // Try each combination
        try_rbf!(Vec<f64>, Vec<f64>, RbfKind::SS);
        try_rbf!(Vec<f64>, Vec<Vec<f64>>, RbfKind::SV);
        try_rbf!(Vec<Vec<f64>>, Vec<f64>, RbfKind::VS);
        try_rbf!(Vec<Vec<f64>>, Vec<Vec<f64>>, RbfKind::VV);

        Err(PyValueError::new_err(
            "Unsupported input shapes for x and y",
        ))
    }

    pub fn predict(&self, x_new: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        Python::with_gil(|py| match &self.inner {
            RbfKind::SS(model) => {
                let x: Vec<f64> = x_new.extract()?;
                model
                    .predict(&x)
                    .map(|out| out.into_py(py))
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
            }
            RbfKind::SV(model) => {
                let x: Vec<f64> = x_new.extract()?;
                model
                    .predict(&x)
                    .map(|out| out.into_py(py))
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
            }
            RbfKind::VS(model) => {
                let x: Vec<Vec<f64>> = x_new.extract()?;
                model
                    .predict(&x)
                    .map(|out| out.into_py(py))
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
            }
            RbfKind::VV(model) => {
                let x: Vec<Vec<f64>> = x_new.extract()?;
                model
                    .predict(&x)
                    .map(|out| out.into_py(py))
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
            }
        })
    }

    pub fn __repr__(&self) -> String {
        match &self.inner {
            RbfKind::SS(m) => format!("Rbf<f64, f64>: epsilon = {}", m.epsilon),
            RbfKind::SV(m) => format!("Rbf<f64, Vec<f64>>: epsilon = {}", m.epsilon),
            RbfKind::VS(m) => format!("Rbf<Vec<f64>, f64>: epsilon = {}", m.epsilon),
            RbfKind::VV(m) => format!("Rbf<Vec<f64>, Vec<f64>>: epsilon = {}", m.epsilon),
        }
    }
}

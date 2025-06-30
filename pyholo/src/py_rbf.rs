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
                    return Rbf::new(x, y, kernel, Some(epsilon))
                        .map($variant)
                        .map(|inner| PyRbf { inner })
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
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            }
            RbfKind::SV(model) => {
                let x: Vec<f64> = x_new.extract()?;
                model
                    .predict(&x)
                    .map(|out| out.into_py(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            }
            RbfKind::VS(model) => {
                let x: Vec<Vec<f64>> = x_new.extract()?;
                model
                    .predict(&x)
                    .map(|out| out.into_py(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            }
            RbfKind::VV(model) => {
                let x: Vec<Vec<f64>> = x_new.extract()?;
                model
                    .predict(&x)
                    .map(|out| out.into_py(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
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

// /// Simple Rbf model.
// #[derive(Clone, Debug)]
// #[pyclass(name = "Rbf")]
// pub struct PyRbf {
//     pub inner: Rbf<Vec<f64>, Vec<f64>>,
// }

// #[pymethods]
// impl PyRbf {
//     /// Instantiates a new `PyRbf` instance.
//     ///
//     /// # Arguments
//     /// * `x`: A n*m matrix containing the training data points.
//     /// * `y`: A n vector containing the corresponding training output values.
//     /// * `kernel`: An optional function that computes the kernel function value.
//     ///             Will default to Gaussian kernel if `None` given.
//     /// * `epsilon`: An optional bandwidth parameter for the kernel.
//     ///              Defaults to 1. if `None` given.
//     ///
//     /// # Returns
//     /// A new `Rbf` instance.
//     #[new]
//     pub fn new(
//         x: &Bound<'_, PyAny>,
//         y: &Bound<'_, PyAny>,
//         kernel_name: Option<&str>,
//         epsilon: Option<f64>,
//     ) -> PyResult<Self> {
//         let kernel_name = kernel_name.unwrap_or("gaussian");
//         let kernel = match kernel_name {
//             "gaussian" => gaussian_kernel,
//             "multiquadric" => multiquadric_kernel,
//             "inverse_multiquadratic" => inverse_multi_kernel,
//             "linear" => linear_kernel,
//             "cubic" => cubic_kernel,
//             "thin_plate_spline" => thin_plate_spline_kernel,
//             _ => {
//                 return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                     "Unsupported kernel type",
//                 ))
//             }
//         };

//         let x_vec: Vec<Vec<f64>> = x.extract()?;
//         let y_vec: Vec<Vec<f64>> = y.extract()?;

//         let inner = Rbf::new(x_vec, y_vec, Some(kernel), epsilon).expect("Failed to create Rbf");

//         Ok(PyRbf { inner })
//     }

//     pub fn predict(&self, x_new: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
//         let x_new: Vec<Vec<f64>> = x_new.extract()?;
//         self.inner
//             .predict(&x_new)
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
//     }

//     pub fn __repr__(&self) -> String {
//         let mut ss = format!(
//             "┌{}┐\n│{: <48}│\n╞{}╡\n│{: <48}│\n",
//             "─".repeat(48),
//             "Rbf Model:",
//             "═".repeat(48),
//             format!("Epsilon: {}", self.inner.epsilon)
//         );
//         ss += &format!("└{}┘\n", "─".repeat(48));
//         ss
//     }
// }

// impl Interpolator<Vec<f64>, Vec<f64>> for PyRbf {
//     fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
//         self.inner.predict(x_new)
//     }
// }

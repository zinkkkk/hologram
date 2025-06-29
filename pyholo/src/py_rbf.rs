#[allow(non_local_definitions)]
use pyo3::prelude::*;

use hologram::{
    kernels::{
        cubic_kernel, gaussian_kernel, inverse_multi_kernel, linear_kernel, multiquadric_kernel,
        thin_plate_spline_kernel,
    },
    rbf::Rbf,
    Interpolator,
};

/// Simple Rbf model.
#[derive(Clone, Debug)]
#[pyclass(name = "Rbf")]
pub struct PyRbf {
    pub inner: Rbf<Vec<f64>, Vec<f64>>,
}

#[pymethods]
impl PyRbf {
    /// Instantiates a new `PyRbf` instance.
    ///
    /// # Arguments
    /// * `x`: A n*m matrix containing the training data points.
    /// * `y`: A n vector containing the corresponding training output values.
    /// * `kernel`: An optional function that computes the kernel function value.
    ///             Will default to Gaussian kernel if `None` given.
    /// * `epsilon`: An optional bandwidth parameter for the kernel.
    ///              Defaults to 1. if `None` given.
    ///
    /// # Returns
    /// A new `Rbf` instance.
    #[new]
    pub fn new(
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        kernel_name: Option<&str>,
        epsilon: Option<f64>,
    ) -> PyResult<Self> {
        let kernel_name = kernel_name.unwrap_or("gaussian");
        let kernel = match kernel_name {
            "gaussian" => gaussian_kernel,
            "multiquadric" => multiquadric_kernel,
            "inverse_multiquadratic" => inverse_multi_kernel,
            "linear" => linear_kernel,
            "cubic" => cubic_kernel,
            "thin_plate_spline" => thin_plate_spline_kernel,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported kernel type",
                ))
            }
        };

        let x_vec: Vec<Vec<f64>> = x.extract()?;
        let y_vec: Vec<Vec<f64>> = y.extract()?;

        let inner = Rbf::new(x_vec, y_vec, Some(kernel), epsilon).expect("Failed to create Rbf");

        Ok(PyRbf { inner })
    }

    pub fn predict(&self, x_new: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let x_new: Vec<Vec<f64>> = x_new.extract()?;
        self.inner
            .predict(&x_new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn __repr__(&self) -> String {
        let mut ss = format!(
            "┌{}┐\n│{: <48}│\n╞{}╡\n│{: <48}│\n|{: <48}│\n",
            "─".repeat(48),
            "Rbf Model:",
            "═".repeat(48),
            "Kernel: Gaussian",
            format!("Epsilon: {}", self.inner.epsilon)
        );
        ss += &format!("└{}┘\n", "─".repeat(48));
        ss
    }
}

impl Interpolator<Vec<f64>, Vec<f64>> for PyRbf {
    fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        self.inner.predict(x_new)
    }
}

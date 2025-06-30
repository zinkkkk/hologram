use crate::{
    kernels::gaussian_kernel, linear_algebra::build_design_matrix, normalisation::{denormalise_data, normalise_data, normalise_data_with, Normalisable}, numeric::Numeric, Interpolator
};

#[cfg(not(any(feature = "openblas", feature = "intel-mkl")))]
use crate::linear_algebra::lu_linear_solver;

#[cfg(any(feature = "openblas", feature = "intel-mkl"))]
use crate::linear_algebra::ndarray_linear_solver;

/// Rbf model.
#[derive(Debug, Clone)]
pub struct Rbf<X, Y> {
    /// Input points.
    pub x: Vec<X>,
    /// Output points.
    pub y: Vec<Y>,
    /// Kernel function.
    kernel: fn(f64, f64) -> f64,
    /// Kernel bandwidth parameter.
    pub epsilon: f64,
    /// Interpolation coefficients.
    pub weights: Vec<Y>,
}

impl<X, Y> Rbf<X, Y>
where
    X: Numeric + PartialEq,
    Y: Numeric,
{
    /// Instantiates a new `Rbf` instance.
    ///
    /// # Arguments
    /// * `x`: A vector of type X containing the training data points.
    /// * `y`: A vector of type Y containing the corresponding training output values.
    /// * `kernel`: An optional function that computes the kernel function value.
    ///             Will default to Gaussian kernel if `None` given.
    /// * `epsilon`: An optional bandwidth parameter for the kernel.
    ///              Defaults to 1. if `None` given.
    ///
    /// # Returns
    /// A new `Rbf` instance.
    pub fn new(
        x: Vec<X>,
        y: Vec<Y>,
        kernel: Option<fn(f64, f64) -> f64>,
        epsilon: Option<f64>,
    ) -> Result<Self, String> {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");

        let kernel = kernel.unwrap_or(gaussian_kernel);
        let epsilon = epsilon.unwrap_or(1.);

        let design_matrix = build_design_matrix(&x, &x, &kernel, epsilon);

        // Solve for weights
        let weights = {
            #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
            {
                ndarray_linear_solver(&design_matrix, &y)?
            }
            #[cfg(not(any(feature = "openblas", feature = "intel-mkl")))]
            {
                lu_linear_solver(&design_matrix, &y)?
            }
        };

        Ok(Rbf {
            x,
            y,
            kernel,
            epsilon,
            weights,
        })
    }
}

impl<X, Y> Interpolator<X, Y> for Rbf<X, Y>
where
    X: Numeric,
    Y: Numeric,
{
    fn predict(&self, x_new: &[X]) -> Result<Vec<Y>, String> {
        let mut result = Y::zeros(x_new.len(), &self.y.first().unwrap());

        for (n, x_n) in x_new.iter().enumerate() {
            for (i, x_i) in self.x.iter().enumerate() {
                let dist = x_n.squared_distance(x_i);
                let kernel_val = (self.kernel)(dist.max(f64::EPSILON), self.epsilon);
                result[n].add_assign(&self.weights[i].multiply_scalar(kernel_val));
            }

            if result[n].is_instance_nan() {
                return Err(String::from("NaN values in output"));
            }
        }

        Ok(result)
    }
}


/// A wrapper around `Rbf` that automatically normalises input and output data using Z-score normalisation.
#[derive(Debug, Clone)]
pub struct RbfNorm<X, Y> {
    /// The underlying Rbf model, which operates on normalised data.
    inner: Rbf<X, Y>,
    /// Mean of the original input data.
    x_mean: X,
    /// Standard deviation of the original input data.
    x_std: X,
    /// Mean of the original output data.
    y_mean: Y,
    /// Standard deviation of the original output data.
    y_std: Y,
}

impl<X, Y> RbfNorm<X, Y>
where
    X: Normalisable + Numeric + PartialEq,
    Y: Normalisable + Numeric,
{
    /// Creates a new `RbfNorm` instance by normalising the input and output data and constructing the underlying RBF model.
    ///
    /// # Arguments
    /// * `x` - Unnormalised input data.
    /// * `y` - Unnormalised output data.
    /// * `kernel` - Optional kernel function.
    /// * `epsilon` - Optional epsilon parameter.
    ///
    /// # Returns
    /// A new `RbfNorm` instance.
    pub fn new(
        x: Vec<X>,
        y: Vec<Y>,
        kernel: Option<fn(f64, f64) -> f64>,
        epsilon: Option<f64>,
    ) -> Result<Self, String> {
        let (x_normalised, x_mean, x_std) = normalise_data(&x);
        let (y_normalised, y_mean, y_std) = normalise_data(&y);

        let inner = Rbf::new(x_normalised, y_normalised, kernel, epsilon)?;

        Ok(Self {
            inner,
            x_mean,
            x_std,
            y_mean,
            y_std,
        })
    }
}

impl<X, Y> Interpolator<X, Y> for RbfNorm<X, Y>
where
    X: Normalisable + Numeric,
    Y: Normalisable + Numeric,
{
    fn predict(&self, x_new: &[X]) -> Result<Vec<Y>, String> {
        let x_normalised = normalise_data_with(x_new, &self.x_mean, &self.x_std);
        let y_normalised = self.inner.predict(&x_normalised)?;
        let y = denormalise_data(&y_normalised, &self.y_mean, &self.y_std);
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_predict_rbf_x_nd_y_1d() {
        let rbf = Rbf::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![5.0, 6.0],
            None,
            None,
        )
        .unwrap();

        let x_new = vec![vec![2.5, 3.5]];
        let prediction = rbf.predict(&x_new).unwrap();

        // Compare the predicted result with the expected result
        assert_abs_diff_eq!(prediction[0], 5.11861403058931, epsilon = 1e-8);
    }

    #[test]
    fn test_predict_rbf_x_1d_y_1d() {
        let rbf =
            Rbf::<f64, f64>::new(vec![1.0, 2.0, 3.0], vec![5.0, 6.0, 8.0], None, None).unwrap();

        let x_new = vec![2.5, 2.7];
        let prediction = rbf.predict(&x_new).unwrap();

        // Compare the predicted result with the expected result
        assert_abs_diff_eq!(prediction[0], 7.240911017466877, epsilon = 1e-8);
        assert_abs_diff_eq!(prediction[1], 7.680299569616757, epsilon = 1e-8);
    }

    #[test]
    fn test_rbf_predict_x_1d_y_3d() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let rbf = Rbf::<f64, [f64; 3]>::new(x, y, None, None).unwrap();

        let x_new = vec![1.5, 2.5];
        let predictions = rbf.predict(&x_new).expect("Prediction failed");

        assert_abs_diff_eq!(predictions[0][0], 2.1326701587635037, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[0][1], 3.1497053968910373, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[0][2], 4.166740635018571, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[1][0], 6.003611746256762, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[1][1], 7.020646984384296, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[1][2], 8.03768222251183, epsilon = 1e-8);
    }
}

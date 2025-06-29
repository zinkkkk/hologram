use crate::{
    kernels::gaussian_kernel,
    linear_algebra::build_design_matrix,
    normalisation::{normalise_data, Normalisable},
    numeric::Numeric,
    Interpolator,
};

#[cfg(not(any(feature = "openblas", feature = "intel-mkl")))]
use crate::linear_algebra::lu_linear_solver;

#[cfg(any(feature = "openblas", feature = "intel-mkl"))]
use crate::linear_algebra::ndarray_linear_solver;

// #[derive(Debug, Clone)]
// pub struct Rbf1 {
//     /// Input points.
//     x: Vec<Vec<f64>>,
//     /// Output points.
//     y: Vec<f64>,
//     /// Kernel function.
//     kernel: fn(f64, f64) -> f64,
//     /// Kernel bandwidth parameter.
//     pub epsilon: f64,
//     /// Interpolation coefficients.
//     weights: Vec<f64>,
// }

// impl Rbf1 {
//     // Calculate the default epsilon based on the bounding hypercube (similar to scipy's approach)
//     fn calculate_default_epsilon(x: &[Vec<f64>]) -> f64 {
//         if x.is_empty() || x[0].is_empty() {
//             return 1.0;
//         }

//         let n_dims = x[0].len();
//         let n_points = x.len();

//         // Find min and max for each dimension
//         let mut min_vals = vec![f64::INFINITY; n_dims];
//         let mut max_vals = vec![-f64::INFINITY; n_dims];

//         for point in x {
//             for (i, &val) in point.iter().enumerate() {
//                 if val < min_vals[i] {
//                     min_vals[i] = val;
//                 }
//                 if val > max_vals[i] {
//                     max_vals[i] = val;
//                 }
//             }
//         }

//         // Calculate edge lengths
//         let edges: Vec<f64> = min_vals
//             .iter()
//             .zip(max_vals.iter())
//             .map(|(min_val, max_val)| max_val - min_val)
//             .filter(|&e| e > 0.0) // Filter out zero-length edges
//             .collect();

//         if edges.is_empty() {
//             return 1.0;
//         }

//         // Calculate epsilon as (product of edges / n_points) ^ (1/num_edges)
//         let product: f64 = edges.iter().product();
//         (product / n_points as f64).powf(1.0 / edges.len() as f64)
//     }

//     pub fn new(
//         x: Vec<Vec<f64>>,
//         y: Vec<f64>,
//         kernel: Option<fn(f64, f64) -> f64>,
//         epsilon: Option<f64>,
//     ) -> Self {
//         assert!(
//             !x.is_empty() && !x[0].is_empty(),
//             "Input data cannot be empty"
//         );
//         assert_eq!(x.len(), y.len(), "Number of x and y points must match");

//         // Use provided epsilon or calculate default based on data
//         let epsilon = epsilon.unwrap_or_else(|| Self::calculate_default_epsilon(&x));

//         // Use provided kernel or default to gaussian
//         let kernel = kernel.unwrap_or(gaussian_kernel);

//         // Build design matrix
//         let design_matrix = build_design_matrix(&x, &x, &kernel, epsilon);

//         // Solve for weights
//         let weights = match lu_linear_solver(&design_matrix, &y) {
//             Ok(weights) => weights,
//             Err(msg) => panic!("Failed to solve linear system: {}", msg),
//         };

//         Rbf1 {
//             x: x.clone(),
//             y: y.clone(),
//             kernel,
//             epsilon,
//             weights,
//         }
//     }
// }

// impl Interpolator<Vec<f64>, f64> for Rbf1 {
//     fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<f64>, String> {
//         if x_new.is_empty() {
//             return Ok(Vec::new());
//         }

//         if x_new[0].len() != self.x[0].len() {
//             return Err(format!(
//                 "Dimensionality mismatch: expected {}, got {}",
//                 self.x[0].len(),
//                 x_new[0].len()
//             ));
//         }

//         let n = self.x.len();
//         let mut result = vec![0.0; x_new.len()];

//         for (k, x) in x_new.iter().enumerate() {
//             for i in 0..n {
//                 let dist = squared_euclidean_distance(x, &self.x[i]);
//                 result[k] += self.weights[i] * (self.kernel)(dist, self.epsilon);
//             }

//             if result[k].is_nan() {
//                 return Err(String::from("NaN values in output"));
//             }
//         }

//         Ok(result)
//     }
// }

// pub fn squared_euclidean_distance(x0: &[f64], x1: &[f64]) -> f64 {
//     x0.iter()
//         .zip(x1.iter())
//         .map(|(&a, &b)| (a - b).powi(2))
//         .sum()
// }

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

    // For normalisation
    /// Input standard deviation.
    x_std: X,
    /// Mean output.
    y_mean: Y,
    /// Output standard deviation.
    y_std: Y,
}

impl<X, Y> Rbf<X, Y>
where
    X: Normalisable + Numeric + PartialEq,
    Y: Normalisable + Numeric,
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

        // Normalise the data for the design matrix
        let (_, x_std, normalised_x) = normalise_data(&x);
        let (y_mean, y_std, normalised_y) = normalise_data(&y);

        let design_matrix = build_design_matrix(&normalised_x, &normalised_x, &kernel, epsilon);

        // Solve for weights
        let weights = {
            #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
            {
                ndarray_linear_solver(&design_matrix, &normalised_y)?
            }
            #[cfg(not(any(feature = "openblas", feature = "intel-mkl")))]
            {
                lu_linear_solver(&design_matrix, &normalised_y)?
            }
        };

        Ok(Rbf {
            x, // un-normalised
            y, // un-normalised
            kernel,
            epsilon,
            weights,
            x_std,
            y_mean,
            y_std,
        })
    }
}

impl<X, Y> Interpolator<X, Y> for Rbf<X, Y>
where
    X: Normalisable,
    Y: Normalisable + Numeric,
{
    fn predict(&self, x_new: &[X]) -> Result<Vec<Y>, String> {
        let mut result = Y::zeros(x_new.len(), &self.y.first().unwrap());

        for (n, x_n) in x_new.iter().enumerate() {
            for (i, x_i) in self.x.iter().enumerate() {
                let dist = x_n.squared_distance(x_i, &self.x_std).max(f64::EPSILON);
                let kernel_val = (self.kernel)(dist, self.epsilon);
                result[n].add_assign(&self.weights[i].multiply_scalar(kernel_val));
            }

            // De-normalise result
            result[n] = result[n].denormalise(&self.y_mean, &self.y_std);
            if result[n].is_instance_nan() {
                return Err(String::from("NaN values in output"));
            }
        }

        Ok(result)
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
        assert_abs_diff_eq!(prediction[0], 5.84298272702809, epsilon = 1e-8);
    }

    #[test]
    fn test_predict_rbf_x_1d_y_1d() {
        let rbf =
            Rbf::<f64, f64>::new(vec![1.0, 2.0, 3.0], vec![5.0, 6.0, 8.0], None, None).unwrap();

        let x_new = vec![2.5, 2.7];
        let prediction = rbf.predict(&x_new).unwrap();

        // Compare the predicted result with the expected result
        assert_abs_diff_eq!(prediction[0], 7.19762989994139, epsilon = 1e-8);
        assert_abs_diff_eq!(prediction[1], 7.609508677512894, epsilon = 1e-8);
    }

    #[test]
    fn test_rbf_predict_x_1d_y_vec3d() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let rbf = Rbf::<f64, [f64; 3]>::new(x, y, None, None).unwrap();

        let x_new = vec![1.5, 2.5];
        let predictions = rbf.predict(&x_new).expect("Prediction failed");

        assert_abs_diff_eq!(predictions[0][0], 1.9666208588631418, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[0][1], 2.9666208588631418, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[0][2], 3.9666208588631418, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[1][0], 6.033379141136859, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[1][1], 7.033379141136859, epsilon = 1e-8);
        assert_abs_diff_eq!(predictions[1][2], 8.033379141136859, epsilon = 1e-8);
    }
}

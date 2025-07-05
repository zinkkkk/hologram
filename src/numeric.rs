use std::fmt::Debug;

/// A trait for numeric types that can be used with the RBF interpolator.
///
/// This trait defines the basic arithmetic operations required for RBF interpolation,
/// along with type conversion and utility methods. Implementations are provided
/// for common numeric types like `f64`, `[f64; 3]`, and `Vec<f64>`.
pub trait Numeric: Default + Debug + Clone {
    fn is_instance_nan(&self) -> bool;
    fn zero(shape: &Self) -> Self;
    fn squared_distance(&self, other: &Self) -> f64;
    fn multiply_scalar(&self, scalar: f64) -> Self;
    fn add_assign(&mut self, other: &Self);
    fn subtract(&self, other: &Self) -> Self;
    fn divide_scalar(&self, scalar: f64) -> Self;

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn to_flattened(&self) -> Vec<f64>;

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn from_flattened(flattened: Vec<f64>) -> Result<Self, String>
    where
        Self: Sized;
}

impl Numeric for f64 {
    fn is_instance_nan(&self) -> bool {
        self.is_nan()
    }

    fn zero(_shape: &Self) -> Self {
        0.
    }

    fn squared_distance(&self, other: &Self) -> f64 {
        (self - other).powi(2)
    }

    fn multiply_scalar(&self, scalar: f64) -> Self {
        self * scalar
    }

    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }

    fn subtract(&self, other: &Self) -> Self {
        self - other
    }

    fn divide_scalar(&self, scalar: f64) -> Self {
        self / scalar
    }

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn to_flattened(&self) -> Vec<f64> {
        vec![*self]
    }

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn from_flattened(mut flattened: Vec<f64>) -> Result<Self, String> {
        if flattened.len() != 1 {
            return Err("Expected exactly one element for f64".to_string());
        }
        Ok(flattened.remove(0))
    }
}

impl Numeric for [f64; 3] {
    fn is_instance_nan(&self) -> bool {
        self.iter().any(|&x| x.is_nan())
    }

    fn zero(_shape: &Self) -> Self {
        [0.0; 3]
    }

    fn squared_distance(&self, other: &Self) -> f64 {
        let dx = self[0] - other[0];
        let dy = self[1] - other[1];
        let dz = self[2] - other[2];
        dx * dx + dy * dy + dz * dz
    }

    fn multiply_scalar(&self, scalar: f64) -> Self {
        let mut result = [0.0; 3];
        for i in 0..3 {
            result[i] = self[i] * scalar;
        }
        result
    }

    fn add_assign(&mut self, other: &Self) {
        for i in 0..3 {
            self[i] += other[i];
        }
    }

    fn subtract(&self, other: &Self) -> Self {
        let mut result = [0.0; 3];
        for i in 0..3 {
            result[i] = self[i] - other[i];
        }
        result
    }

    fn divide_scalar(&self, scalar: f64) -> Self {
        let mut result = [0.0; 3];
        for i in 0..3 {
            result[i] = self[i] / scalar;
        }
        result
    }

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn to_flattened(&self) -> Vec<f64> {
        self.to_vec()
    }

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn from_flattened(flattened: Vec<f64>) -> Result<Self, String> {
        if flattened.len() != 3 {
            return Err("Expected exactly 3 elements for [f64; 3]".to_string());
        }
        Ok([flattened[0], flattened[1], flattened[2]])
    }
}

impl Numeric for Vec<f64> {
    fn is_instance_nan(&self) -> bool {
        self.iter().any(|&x| x.is_nan())
    }

    fn zero(shape: &Self) -> Self {
        vec![0.0; shape.len()]
    }

    fn squared_distance(&self, other: &Self) -> f64 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }

    fn multiply_scalar(&self, scalar: f64) -> Self {
        self.iter().map(|&x| x * scalar).collect()
    }

    fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a += b;
        }
    }

    fn subtract(&self, other: &Self) -> Self {
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }

    fn divide_scalar(&self, scalar: f64) -> Self {
        self.iter().map(|&x| x / scalar).collect()
    }

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn to_flattened(&self) -> Vec<f64> {
        self.clone()
    }

    #[cfg(any(feature = "openblas", feature = "intel-mkl"))]
    fn from_flattened(flattened: Vec<f64>) -> Result<Self, String> {
        Ok(flattened)
    }
}

/// Calculates the root-mean-square error (RMSE) between two vectors.
///
/// # Arguments
/// * `y_pred`: A vector of predicted output values.
/// * `y_actual`: A vector of actual output values.
///
/// # Returns
/// The RMSE between the two vectors of output values.
pub fn calculate_rmse<T>(predictions: &[T], targets: &[T]) -> f64
where
    T: Numeric,
{
    let mse = predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| pred.squared_distance(target))
        .sum::<f64>()
        / predictions.len() as f64;
    mse.sqrt()
}

/// Generates `n` evenly spaced values between `start` and `end`, inclusive.
///
/// This function is equivalent to NumPy's `linspace(start, end, num)`
/// and is commonly used for sampling a 1D range of values.
///
/// # Arguments
/// * `start` - The starting value of the sequence.
/// * `end` - The final value of the sequence.
/// * `n` - The number of values to generate. Must be >= 2.
///
/// # Returns
/// A `Vec<f64>` containing `n` evenly spaced values from `start` to `end`.
pub fn linspace(start: &f64, end: &f64, n: usize) -> Vec<f64> {
    assert!(n >= 2, "linspace requires n >= 2");
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

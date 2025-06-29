pub mod kernels;
pub mod lagrangian;
pub mod linear;
pub mod linear_algebra;
pub mod normalisation;
pub mod numeric;
pub mod rbf;
pub mod utilities;

/// Can interpolate
///
/// Supported `X`, `Y` combinations currently:
/// Rbf: `X` can be: Vec<f64>, [f64;3], f64
///      `Y` can be: [f64;3], f64
/// Linear: `X` can be: f64,
///         `Y` can be: f64
pub trait Interpolator<X, Y> {
    /// Predict the output `y_new` given points `x_new`.
    ///
    /// # Arguments
    /// * `x_new`: New input data points to predict the output values for.
    ///
    /// # Returns
    /// A `Result` containing output values if the computation is successful.
    fn predict(&self, x_new: &[X]) -> Result<Vec<Y>, String>;
}

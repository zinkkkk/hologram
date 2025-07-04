use approx::assert_abs_diff_eq;
use hologram::{kernels::gaussian_kernel, numeric::linspace, rbf::Rbf, Interpolator};
use std::f64::consts::PI;
use std::time::Instant;

fn blackbox(pt: f64) -> f64 {
    let sum: f64 = pt.powi(2) + 10.0 * (2.0 * PI * pt).cos();
    10.0 + sum
}

fn main() -> Result<(), String> {
    let start_time = Instant::now();

    // Parameters
    let bounds = (-5.0, 5.0);
    let n_points = 2000;

    // Generate training points
    let x_train: Vec<f64> = linspace(&bounds.0, &bounds.1, n_points);

    // Generate target values using the blackbox function
    let y_train: Vec<f64> = x_train.iter().map(|x| blackbox(*x)).collect();

    let rbf = Rbf::new(x_train, y_train, Some(gaussian_kernel), Some(1.0))?;

    // Generate test points
    let n_test = 1000;
    let x_test: Vec<f64> = linspace(&bounds.0, &bounds.1, n_test);

    // Make predictions
    let y_pred = rbf.predict(&x_test)?;

    println!("Predicted values: {:?}", y_pred[0]);

    let elapsed = start_time.elapsed();
    println!("Execution time: {:.2?}", elapsed);

    assert_abs_diff_eq!(y_pred[1], 44.85706302848141, epsilon = 1e-6);

    Ok(())
}

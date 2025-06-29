use hologram::kernels::gaussian_kernel;
use hologram::rbf::Rbf;
use hologram::Interpolator;
use std::f64::consts::PI;

fn blackbox(pts: &[f64], dimension: usize) -> Vec<f64> {
    let sum: f64 = pts
        .iter()
        .map(|&x| x.powi(2) + 10.0 * (2.0 * PI * x).cos())
        .sum();
    vec![10.0 * dimension as f64 + sum]
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

fn main() {
    // Parameters
    let bounds = (-5.0, 5.0);
    let n_points = 5;
    let dimension = 1;

    // Generate training points
    let x_train: Vec<Vec<f64>> = linspace(bounds.0, bounds.1, n_points)
        .into_iter()
        .map(|x| vec![x]) // Make it 2D: Vec<Vec<f64>>
        .collect();

    // Generate target values using the blackbox function
    let y_train: Vec<Vec<f64>> = x_train.iter().map(|x| blackbox(x, dimension)).collect();

    println!("Training inputs:\n{:?}", x_train);
    println!("Training targets:\n{:?}", y_train);

    let rbf = Rbf::new(x_train.clone(), y_train, Some(gaussian_kernel), Some(1.0))
        .expect("Failed to create RBF");

    // Generate test points
    let n_test = 9;
    let x_test: Vec<Vec<f64>> = linspace(bounds.0, bounds.1, n_test)
        .into_iter()
        .map(|x| vec![x]) // Make it 2D: Vec<Vec<f64>>
        .collect();

    // Make predictions
    let y_pred = rbf.predict(&x_test).expect("Prediction failed");

    println!("Test inputs:\n{:?}", x_test);
    println!("Test targets:\n{:?}", y_pred);
}

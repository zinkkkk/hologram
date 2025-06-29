/// kernels (All kernels accept squared Euclidean for the `distance`)
/// Guassian
pub fn gaussian_kernel(distance: f64, bandwidth: f64) -> f64 {
    (-0.5 * distance / (bandwidth * bandwidth)).exp()
}

/// Multiquadratic
pub fn multiquadric_kernel(distance: f64, bandwidth: f64) -> f64 {
    (distance + bandwidth * bandwidth).sqrt()
}

/// Inverse
pub fn inverse_multi_kernel(distance: f64, bandwidth: f64) -> f64 {
    1.0 / multiquadric_kernel(distance, bandwidth)
}

/// Linear
pub fn linear_kernel(distance: f64, _bandwidth: f64) -> f64 {
    distance
}

/// Cubic
pub fn cubic_kernel(distance: f64, _bandwidth: f64) -> f64 {
    distance.powi(2)
}

/// Polynomial
pub fn polynomial_kernel(distance: f64, bandwidth: f64) -> f64 {
    (distance + 1.0).powi(bandwidth as i32)
}

/// Thin plate
pub fn thin_plate_spline_kernel(distance: f64, _bandwidth: f64) -> f64 {
    if distance == 0.0 {
        0.0
    } else {
        distance.powi(2) * distance.ln()
    }
}

/// Exponential
pub fn exponential_kernel(distance: f64, bandwidth: f64) -> f64 {
    (-distance / bandwidth).exp()
}

/// Neural tangent
pub fn neural_tangent_kernel(distance: f64, bandwidth: f64) -> f64 {
    let norm_distance = distance / bandwidth;
    let theta = norm_distance.acos();
    (1.0 - norm_distance * theta.sin() + norm_distance.powi(2) * (1.0 - theta.cos()))
        .max(f64::EPSILON)
}

/// Rational quadratic
pub fn rational_quadratic_kernel(distance: f64, alpha: f64) -> f64 {
    (1.0 + (distance.powi(2) / (2.0 * alpha))).powf(-alpha)
}

/// Cauchy
pub fn cauchy_kernel(distance: f64, sigma: f64) -> f64 {
    1.0 / (1.0 + (distance.powi(2) / sigma.powi(2)))
}

/// Laplacian
pub fn laplacian_kernel(distance: f64, gamma: f64) -> f64 {
    (-gamma * distance.abs()).exp()
}

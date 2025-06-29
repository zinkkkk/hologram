use std::f64::consts::PI;

use crate::Interpolator;

/// Represents a Lagrangian interpolator, used to perform polynomial interpolation based on a set of
/// input and output points.
///
/// # Fields
/// * `x`: A vector of input points where interpolation is performed.
/// * `y`: A vector of output points corresponding to the input points,
///        where each output point is a 3-element array [x, y, z].
/// * `denominators`: Precomputed denominators used in the Lagrangian interpolation formula to
///                   optimise the calculation.
pub struct Lagrangian {
    x: Vec<f64>,
    y: Vec<[f64; 3]>,
    denominators: Vec<f64>,
}

impl Lagrangian {
    pub fn new(x: Vec<f64>, y: Vec<[f64; 3]>, chebyshev: bool) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");

        // Chebyshev nodes
        let n = x.len();
        let x = if chebyshev {
            let a = *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let b = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            (0..n)
                .map(|i| {
                    let cos_value = ((2 * i + 1) as f64 * PI / (2.0 * n as f64)).cos();
                    0.5 * ((b - a) * cos_value + (b + a))
                })
                .collect::<Vec<f64>>()
        } else {
            x
        };

        // Compute denominators in advance
        let mut denominators = vec![1.0; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    denominators[i] *= x[i] - x[j]; //.max(std::f64::EPSILON)
                }
            }
            if denominators[i] == 0.0 {
                panic!("Division by zero in basis function calculation.");
            }
        }

        Lagrangian { x, y, denominators }
    }

    fn compute_basis(&self, x_n: f64, i: usize) -> f64 {
        let mut basis = 1.0;
        for j in 0..self.x.len() {
            if i != j {
                basis *= x_n - self.x[j];
            }
        }
        basis / self.denominators[i]
    }
}

impl Interpolator<f64, [f64; 3]> for Lagrangian {
    fn predict(&self, x_new: &[f64]) -> Result<Vec<[f64; 3]>, String> {
        let mut results = Vec::with_capacity(x_new.len());

        for &x_n in x_new {
            let mut result = [0.0; 3];

            for i in 0..self.x.len() {
                let basis = self.compute_basis(x_n, i);
                for k in 0..3 {
                    result[k] += self.y[i][k] * basis;
                }
            }

            if result.iter().any(|&val| val.is_nan()) {
                return Err(String::from("NaN values in output"));
            }

            results.push(result);
        }

        Ok(results.into_iter().rev().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagrangian_interpolator() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0]];

        let interpolator = Lagrangian::new(x, y, false);

        let x_new = vec![0.5, 1.5];

        let result = interpolator.predict(&x_new).expect("Failed to predict");

        let expected = vec![[2.875, 2.375, 0.75], [2.375, 1.875, 1.75]];

        assert_eq!(result, expected);
    }
}

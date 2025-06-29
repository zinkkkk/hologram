use crate::{utilities::merge_unique_points, Interpolator};

// Currently only 1d
pub struct Linear {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Linear {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");
        Linear { x, y }
    }

    fn find_segment_binary(&self, x_new: f64) -> Option<(usize, usize)> {
        match self
            .x
            .binary_search_by(|probe| probe.partial_cmp(&x_new).unwrap())
        {
            Ok(index) => {
                if index < self.x.len() - 1 {
                    Some((index, index + 1))
                } else {
                    None
                }
            }
            Err(index) => {
                if index == 0 || index >= self.x.len() {
                    None
                } else {
                    Some((index - 1, index))
                }
            }
        }
    }

    pub fn update(&mut self, x_new: Vec<f64>, y_new: Vec<f64>) {
        assert_eq!(
            x_new.len(),
            y_new.len(),
            "x_new and y_new must have the same length"
        );

        merge_unique_points(&mut self.x, &mut self.y, x_new, y_new);

        let mut combined = self
            .x
            .iter()
            .cloned()
            .zip(self.y.iter().cloned())
            .collect::<Vec<_>>();
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        self.x = combined.iter().map(|&(x, _)| x).collect();
        self.y = combined.iter().map(|&(_, y)| y).collect();
    }
}

impl Interpolator<f64, f64> for Linear {
    fn predict(&self, x_new: &[f64]) -> Result<Vec<f64>, String> {
        let mut result = Vec::with_capacity(x_new.len());

        let x_min = self.x[0];
        let x_max = self.x[self.x.len() - 1];
        let y_min = self.y[0];
        let y_max = self.y[self.y.len() - 1];
        let x_first_diff = self.x[1] - self.x[0];
        let x_last_diff = self.x[self.x.len() - 1] - self.x[self.x.len() - 2];

        for &x_val in x_new {
            if x_val < x_min {
                // Extrapolation to the left
                let y_val = y_min + (x_val - x_min) * (self.y[1] - y_min) / x_first_diff;
                result.push(y_val);
            } else if x_val > x_max {
                // Extrapolation to the right
                let y_val =
                    y_max + (x_val - x_max) * (y_max - self.y[self.y.len() - 2]) / x_last_diff;
                result.push(y_val);
            } else {
                match self.find_segment_binary(x_val) {
                    Some((i, j)) => {
                        let x_i = self.x[i];
                        let x_j = self.x[j];
                        let y_i = self.y[i];
                        let y_j = self.y[j];

                        // Linear interpolation formula
                        let t = (x_val - x_i) / (x_j - x_i);
                        let y_val = y_i + t * (y_j - y_i);
                        result.push(y_val);
                    }
                    None => {
                        return Err(format!("Value {} is out of the interpolation range", x_val));
                    }
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_linear_interpolator_1d() {
        let interpolator = Linear::new(vec![1.0, 2.0, 3.0], vec![5.0, 6.0, 8.0]);

        let x_new = vec![2.5, 2.7];
        let prediction = interpolator.predict(&x_new).unwrap();

        assert_eq!(prediction, vec![7.0, 7.4]);
    }

    #[test]
    fn test_update_linear_interpolator_1d() {
        let mut interpolator = Linear::new(vec![1.0, 2.0, 3.0], vec![1.0, 4.0, 9.0]);

        let x_new = vec![2.5, 3.5];
        let y_new = vec![6.25, 12.25];
        interpolator.update(x_new, y_new);

        assert_eq!(interpolator.x.len(), 5);
        assert_eq!(interpolator.y.len(), 5);

        let x_expected = vec![1.0, 2.0, 2.5, 3.0, 3.5];
        let y_expected = vec![1.0, 4.0, 6.25, 9.0, 12.25];

        for (a, b) in interpolator.x.iter().zip(x_expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        for (a, b) in interpolator.y.iter().zip(y_expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

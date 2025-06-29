/// Normalises a slice of data by calculating its mean and standard deviation,
/// then scaling the data to have zero mean and unit variance.
///
/// # Arguments
/// * `data` - A slice of data points to be normalised.
///
/// # Returns
/// A tuple containing:
/// 1. The mean of the input data
/// 2. The standard deviation of the input data
/// 3. The normalised data as a vector
pub fn normalise_data<T: Normalisable>(data: &[T]) -> (T, T, Vec<T>) {
    let (mean, std_dev) = T::mean_std(data);
    let normalised_data = T::normalise(data, &mean, &std_dev);
    (mean, std_dev, normalised_data)
}
/// A trait for types that can be normalised to have zero mean and unit variance.
///
/// This trait provides methods for calculating statistics and transforming data
/// to and from normalized space. Implementations are provided for common numeric
/// types like `f64`, `[f64; 3]`, and `Vec<f64>`.
pub trait Normalisable {
    fn mean_std(data: &[Self]) -> (Self, Self)
    where
        Self: Sized;

    fn normalise(data: &[Self], mean: &Self, std: &Self) -> Vec<Self>
    where
        Self: Sized;

    fn denormalise(&self, mean: &Self, std_dev: &Self) -> Self;

    fn squared_distance(&self, other: &Self, std_dev: &Self) -> f64;
}

impl Normalisable for f64 {
    fn mean_std(data: &[Self]) -> (Self, Self) {
        let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        (mean, std_dev)
    }

    fn normalise(data: &[Self], mean: &Self, std: &Self) -> Vec<Self> {
        data.iter().map(|&x| (x - mean) / std).collect()
    }

    fn denormalise(&self, mean: &Self, std_dev: &Self) -> Self {
        self * std_dev + mean
    }

    fn squared_distance(&self, other: &Self, std_dev: &Self) -> f64 {
        ((self - other) / std_dev).powi(2)
    }
}

impl Normalisable for [f64; 3] {
    fn mean_std(data: &[Self]) -> (Self, Self) {
        let means = [
            data.iter().map(|x| x[0]).sum::<f64>() / data.len() as f64,
            data.iter().map(|x| x[1]).sum::<f64>() / data.len() as f64,
            data.iter().map(|x| x[2]).sum::<f64>() / data.len() as f64,
        ];

        let variances = [
            data.iter().map(|x| (x[0] - means[0]).powi(2)).sum::<f64>() / data.len() as f64,
            data.iter().map(|x| (x[1] - means[1]).powi(2)).sum::<f64>() / data.len() as f64,
            data.iter().map(|x| (x[2] - means[2]).powi(2)).sum::<f64>() / data.len() as f64,
        ];

        let std_devs = [
            variances[0].sqrt(),
            variances[1].sqrt(),
            variances[2].sqrt(),
        ];

        (means, std_devs)
    }

    fn normalise(data: &[Self], means: &Self, std_devs: &Self) -> Vec<Self> {
        data.iter()
            .map(|&x| {
                [
                    (x[0] - means[0]) / std_devs[0],
                    (x[1] - means[1]) / std_devs[1],
                    (x[2] - means[2]) / std_devs[2],
                ]
            })
            .collect()
    }

    fn denormalise(&self, mean: &Self, std_dev: &Self) -> Self {
        let mut result = [0.0; 3];
        for i in 0..3 {
            result[i] = self[i] * std_dev[i] + mean[i];
        }
        result
    }

    fn squared_distance(&self, other: &Self, std_dev: &Self) -> f64 {
        let dx = (self[0] - other[0]) / std_dev[0];
        let dy = (self[1] - other[1]) / std_dev[1];
        let dz = (self[2] - other[2]) / std_dev[2];
        dx * dx + dy * dy + dz * dz
    }
}

impl Normalisable for Vec<f64> {
    fn mean_std(data: &[Self]) -> (Self, Self) {
        let len = data[0].len();
        let mut means = vec![0.0; len];
        let mut std_devs = vec![0.0; len];

        for i in 0..len {
            let mean = data.iter().map(|x| x[i]).sum::<f64>() / data.len() as f64;
            let variance =
                data.iter().map(|x| (x[i] - mean).powi(2)).sum::<f64>() / data.len() as f64;
            let std_dev = variance.sqrt();
            means[i] = mean;
            std_devs[i] = std_dev;
        }

        (means, std_devs)
    }

    fn normalise(data: &[Self], means: &Self, std_devs: &Self) -> Vec<Self> {
        data.iter()
            .map(|x| {
                x.iter()
                    .enumerate()
                    .map(|(i, &xi)| (xi - means[i]) / std_devs[i])
                    .collect()
            })
            .collect()
    }

    fn denormalise(&self, mean: &Self, std_dev: &Self) -> Self {
        self.iter()
            .zip(mean.iter())
            .zip(std_dev.iter())
            .map(|((a, m), s)| a * s + m)
            .collect()
    }

    fn squared_distance(&self, other: &Self, std_dev: &Self) -> f64 {
        self.iter()
            .zip(other.iter())
            .zip(std_dev.iter())
            .map(|((a, b), c)| ((a - b) / c).powi(2))
            .sum()
    }
}

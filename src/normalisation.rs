/// Z-score normalises a slice of data by calculating its mean and standard deviation,
/// then scaling the data to have zero mean and unit variance.
///
/// # Arguments
/// * `data` - A slice of data points to be normalised.
///
/// # Returns
/// A tuple containing:
/// 1. The normalised data as a vector
/// 2. The mean of the input data
/// 3. The standard deviation of the input data
///
/// # Example
/// ```
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let (normalised, mean, std_dev) = normalise_data(&data);
/// ```
pub fn normalise_data<T: Normalisable>(data: &[T]) -> (Vec<T>, T, T) {
    let (mean, std_dev) = T::mean_std(data);
    let normalised_data = T::normalise(data, &mean, &std_dev);
    (normalised_data, mean, std_dev)
}

/// Normalises a slice of data using the provided mean and standard deviation.
///
/// This performs Z-score normalisation by computing:
/// `x_normalised = (x - mean) / std_dev`
///
/// # Arguments
/// * `data` - A slice of data points to be normalised.
/// * `mean` - The mean to use for normalisation.
/// * `std_dev` - The standard deviation to use for normalisation.
///
/// # Returns
/// A `Vec<T>` containing the normalised data.
pub fn normalise_data_with<T: Normalisable>(data: &[T], mean: &T, std_dev: &T) -> Vec<T> {
    T::normalise(data, mean, std_dev)
}

/// Reverses Z-score normalisation on a slice of normalised data.
///
/// This function takes data that has been normalised using the Z-score method:
/// `x_normalised = (x - mean) / std_dev`, and reconstructs the original values
/// by computing: `x = x_normalised * std_dev + mean`.
///
/// # Arguments
/// * `data` - A slice of normalised data.
/// * `mean` - The mean used during normalisation.
/// * `std_dev` - The standard deviation used during normalisation.
///
/// # Returns
/// A `Vec<T>` containing the denormalised data points.
///
/// # Example
/// ```
/// let (normalised, mean, std_dev) = normalise_data(&original_data);
/// let denormalised = denormalise_data(&normalised, &mean, &std_dev);
/// ```
pub fn denormalise_data<T: Normalisable>(data: &[T], mean: &T, std_dev: &T) -> Vec<T> {
    data.iter().map(|x| x.denormalise(mean, std_dev)).collect()
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
}

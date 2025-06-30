use approx::assert_abs_diff_eq;
use hologram::{
    kernels::thin_plate_spline_kernel,
    normalisation::{denormalise_data, normalise_data, normalise_data_with},
    numeric::linspace,
    rbf::{Rbf, RbfNorm},
    Interpolator,
};

fn main() -> Result<(), String> {
    // Training input: Vec<f64>
    let x_train = vec![
        0.000, 512.000, 1182.490, 1911.273, 2788.547, 4227.750, 6481.706, 9609.367, 11773.210,
        13188.649, 14400.000,
    ];

    // Training output: Vec<[f64; 3]>
    let y_train = vec![
        [6950.000, 0.000, 0.000],
        [5930.999, 4386.866, 974.859],
        [2498.780, 8537.536, 1897.230],
        [-1993.234, 10819.437, 2404.319],
        [-7051.251, 11465.688, 2547.931],
        [-13549.361, 9748.529, 2166.340],
        [-19093.612, 3887.659, 863.924],
        [-18104.032, -5716.279, -1270.284],
        [-11425.229, -10664.783, -2369.952],
        [-4206.775, -11312.308, -2513.846],
        [3167.219, -7987.115, -1774.914],
    ];

    // Predict at 200 new points between min and max of training
    let x_test = linspace(&x_train[0], &x_train.last().unwrap(), 200);

    //##########################################
    // Variation 1: with explicit normalisation
    //##########################################

    // Normalise the data [Not necessary here, but demonstrating]
    let (x_train_normalised, x_mean, x_std) = normalise_data(&x_train);
    let (y_train_normalised, y_mean, y_std) = normalise_data(&y_train);

    // Create RBF interpolator using thin-plate kernel
    let rbf = Rbf::new(
        x_train_normalised,
        y_train_normalised,
        Some(thin_plate_spline_kernel),
        Some(1.0),
    )?;

    // Normalise test input
    let x_test_normalised = normalise_data_with(&x_test, &x_mean, &x_std);

    // Predict
    let y_pred_normalised = rbf.predict(&x_test_normalised)?;

    // Denormalise output
    let y_pred = denormalise_data(&y_pred_normalised, &y_mean, &y_std);

    // Check random predicted values
    println!("Some predicted value: {:?}", y_pred[y_pred.len() - 10]);
    let expected = [-665.4001594540996, -10373.452862411561, -2305.211375109957];
    let actual = y_pred[y_pred.len() - 10];
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, e, epsilon = 1e-6);
    }

    //####################################################
    // Variation 2: with implicit normalisation (RbfNorm)
    //####################################################

    // Create RBF interpolator using thin-plate kernel
    let rbf = RbfNorm::new(x_train, y_train, Some(thin_plate_spline_kernel), Some(1.0))?;

    // Predict
    let y_pred = rbf.predict(&x_test)?;

    // Check random predicted values
    println!("Some predicted value: {:?}", y_pred[y_pred.len() - 10]);
    let expected = [-665.4001594540996, -10373.452862411561, -2305.211375109957];
    let actual = y_pred[y_pred.len() - 10];
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, e, epsilon = 1e-6);
    }

    Ok(())
}

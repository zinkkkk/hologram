use approx::assert_abs_diff_eq;
use hologram::{kernels::thin_plate_spline_kernel, numeric::linspace, rbf::Rbf, Interpolator};
use std::time::Instant;

fn main() -> Result<(), String> {
    let start_time = Instant::now();

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

    // Create RBF interpolator using thin-plate kernel
    let rbf = Rbf::new(x_train, y_train, Some(thin_plate_spline_kernel), Some(1.0))?;

    // Predict
    let y_pred = rbf.predict(&x_test)?;

    // Check random predicted values
    println!("Some predicted value: {:?}", y_pred[y_pred.len() - 10]);
    let expected = [-822.4664346074902, -10299.058145857985, -2288.679220095764];
    let actual = y_pred[y_pred.len() - 10];
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, e, epsilon = 1e-6);
    }

    let elapsed = start_time.elapsed();
    println!("Execution time: {:.2?}", elapsed);

    Ok(())
}

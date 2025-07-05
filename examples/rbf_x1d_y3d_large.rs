// use approx::assert_abs_diff_eq;
use hologram::{kernels::thin_plate_spline_kernel, numeric::linspace, rbf::Rbf, Interpolator};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::Command;
use std::time::Instant;

fn extract_csv_from_tar_xz(path: &str, output_path: &str) -> Result<(), String> {
    // Use system tar to extract the file
    let status = Command::new("tar")
        .args(["-xJf", path, "-C", output_path])
        .status()
        .map_err(|e| format!("Failed to run tar: {}", e))?;

    if !status.success() {
        return Err(format!("tar exited with status: {}", status));
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let archive_path = "examples/orbit_data.tar.xz";
    let output_dir = "examples";

    extract_csv_from_tar_xz(archive_path, output_dir)?;

    let file = File::open("examples/orbit_data.csv")?;
    let reader = BufReader::new(file);

    let mut x_train = Vec::new(); // time
    let mut y_train = Vec::new(); // [x, y, z]

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // skip header
        }

        let fields: Vec<&str> = line.trim().split(',').collect();
        if fields.len() < 4 {
            continue; // skip invalid rows
        }

        let time: f64 = fields[0].parse()?;
        let pos_x: f64 = fields[1].parse()?;
        let pos_y: f64 = fields[2].parse()?;
        let pos_z: f64 = fields[3].parse()?;

        x_train.push(time);
        y_train.push([pos_x, pos_y, pos_z]);
    }

    if x_train.is_empty() || y_train.is_empty() {
        return Err("No data read from file".into());
    }

    // Predict at 200 new evenly spaced time points
    let x_test = linspace(&x_train[0], &x_train[x_train.len() - 1], 200);

    // Create RBF interpolator
    let start_time = Instant::now();
    let rbf = Rbf::new(x_train, y_train, Some(thin_plate_spline_kernel), Some(1.0))?;

    // Predict
    let y_pred = rbf.predict(&x_test)?;

    // Check a random prediction
    let idx = y_pred.len() - 10;
    println!("Some predicted value: {:?}", y_pred[idx]);

    let elapsed = start_time.elapsed();
    println!("Execution time: {:.2?}", elapsed);

    Ok(())
}

use crate::numeric::Numeric;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

#[cfg(any(feature = "openblas", feature = "intel-mkl"))]
use ndarray::{Array1, Array2};
#[cfg(any(feature = "openblas", feature = "intel-mkl"))]
use ndarray_linalg::Solve;

#[cfg(feature = "faer")]
use faer::prelude::*;

/// Solves a linear system Ax = b using LU decomposition.
///
/// # Arguments
/// * `mat` - The coefficient matrix A as a 2D vector of f64 values.
/// * `rhs` - The right-hand side vector b with elements of any numeric type T.
///
/// # Returns
/// * `Ok(Vec<T>)` - The solution vector x if successful.
/// * `Err(String)` - An error message if the system cannot be solved.
pub fn lu_linear_solver<T>(mat: &[Vec<f64>], rhs: &[T]) -> Result<Vec<T>, String>
where
    T: Numeric,
{
    let n = mat.len();
    if n == 0 || n != rhs.len() {
        return Err("Design matrix and RHS must be non-empty and the same length.".to_string());
    }

    // Estimate condition number using row and column sums
    let mut max_row_sum: f64 = 0.0;
    let mut max_col_sum = vec![0.0; mat[0].len()];

    for row in mat {
        let row_sum: f64 = row.iter().map(|x| x.abs()).sum();
        max_row_sum = max_row_sum.max(row_sum);
        for (j, val) in row.iter().enumerate() {
            max_col_sum[j] += val.abs();
        }
    }

    let max_col_sum = max_col_sum.into_iter().fold(0.0, f64::max);
    let cond_est = max_row_sum * max_col_sum;

    if cond_est <= 1e5 {
        // Good conditioning — skip normalization
        return lu_linear_solver_raw(mat, rhs);
    }

    // Normalise matrix rows and RHS
    let mut norm_mat = Vec::with_capacity(n);
    let mut scaled_rhs = Vec::with_capacity(n);

    for (i, row) in mat.iter().enumerate() {
        let norm = row.iter().map(|x| x.abs()).sum::<f64>().max(f64::EPSILON);
        let mut row_out = Vec::with_capacity(row.len());
        for &x in row {
            row_out.push(x / norm);
        }
        norm_mat.push(row_out);
        scaled_rhs.push(rhs[i].divide_scalar(norm));
    }

    // Solve normalized system
    lu_linear_solver_raw(&norm_mat, &scaled_rhs)
}

pub fn lu_linear_solver_raw<T>(mat: &[Vec<f64>], rhs: &[T]) -> Result<Vec<T>, String>
where
    T: Numeric,
{
    let n = mat.len();

    let (lu, p) = match lu_decomposition(mat) {
        LuDecompResult::Success { lu, p } => (lu, p),
        LuDecompResult::Failure(err) => return Err(err),
    };

    // Forward substitution
    let mut y: Vec<T> = Vec::with_capacity(n);
    for i in 0..n {
        let mut sum = T::zero(&rhs[0]);
        for j in 0..i {
            sum.add_assign(&y[j].multiply_scalar(lu[i][j]));
        }
        y.push(rhs[p[i]].subtract(&sum));
    }

    // Backward substitution
    let mut x = vec![T::zero(&y[0]); n];
    for i in (0..n).rev() {
        let mut sum = T::zero(&y[0]);
        for j in i + 1..n {
            sum.add_assign(&x[j].multiply_scalar(lu[i][j]));
        }
        let denom = lu[i][i];
        x[i] = y[i].subtract(&sum).divide_scalar(denom);

        if x[i].is_instance_nan() {
            return Err(format!(
                "NaN detected in backward substitution at index {}",
                i
            ));
        }
    }

    Ok(x)
}

/// Result of an LU decomposition.
///
/// This enum represents the outcome of performing LU decomposition on a matrix.
pub enum LuDecompResult {
    Success { lu: Vec<Vec<f64>>, p: Vec<usize> },
    Failure(String),
}

/// Performs LU decomposition with partial pivoting on a given square matrix.
/// The function decomposes a square matrix `mat` into its lower triangular
/// (L) and upper triangular (U) components, along with a permutation vector `p`
/// to handle row swaps due to partial pivoting.
///
/// # Arguments
/// * `mat` - A reference to a square matrix.
/// * `pivot_epsilon` - A threshold for detecting zero pivot in LU decomposition.
///
/// # Returns
/// A tuple of `lu` and `p` as `Ok((lu, p))`
/// - `lu` is a matrix where the diagonal and above contain elements of U,
///                       and below the diagonal contains elements of L.
/// - `p` is a permutation vector, representing the row swaps applied to `mat`.
///
/// Or an `Err(String)` if the decomposition fails, such as encountering a zero pivot.
pub fn lu_decomposition(mat: &[Vec<f64>]) -> LuDecompResult {
    let n = mat.len();
    let mut lu = mat.to_vec();
    let mut p = (0..n).collect::<Vec<usize>>();

    #[cfg(feature = "rayon")]
    let parallel_threshold = 100;

    for k in 0..n - 1 {
        // Pivot selection (sequential)
        let mut max_row = k;
        for i in k + 1..n {
            if lu[i][k].abs() > lu[max_row][k].abs() {
                max_row = i;
            }
        }

        if lu[max_row][k] == 0.0 {
            return LuDecompResult::Failure(format!(
                "Exact zero pivot encountered at column {}",
                k
            ));
        }

        if max_row != k {
            lu.swap(k, max_row);
            p.swap(k, max_row);
        }

        let pivot = lu[k][k];

        // Split safely to avoid borrow conflicts
        let (head, tail) = lu.split_at_mut(k + 1);
        let pivot_row = &head[k];

        #[cfg(feature = "rayon")]
        {
            if n - k - 1 > parallel_threshold {
                tail.par_iter_mut().for_each(|row| {
                    update_row(row, pivot_row, k, pivot);
                });
                continue;
            }
        }

        // Sequential fallback
        for row in tail.iter_mut() {
            update_row(row, pivot_row, k, pivot);
        }
    }

    LuDecompResult::Success { lu, p }
}

fn update_row(row: &mut [f64], upper_row_k: &[f64], k: usize, pivot: f64) {
    let factor = row[k] / pivot;
    row[k] = factor;
    let row_tail = &mut row[(k + 1)..];
    let lu_k_row_tail = &upper_row_k[(k + 1)..];
    for (j, &lu_kj) in lu_k_row_tail.iter().enumerate() {
        row_tail[j] -= factor * lu_kj;
    }
}

/// Computes the Frobenius norm of a square or rectangular matrix.
///
/// The Frobenius norm is the square root of the sum of the squares of all elements in the matrix.
/// It measures the overall magnitude or "size" of the matrix.
///
/// # Arguments
/// * `mat` - The input matrix as a slice of rows, each row a Vec<f64>.
///
/// # Returns
/// The Frobenius norm as an f64 scalar.
pub fn matrix_frobenius_norm(mat: &[Vec<f64>]) -> f64 {
    mat.iter()
        .flat_map(|row| row.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// Builds a design matrix based on two sets of input vectors and a kernel function.
///
/// # Arguments
/// * `x0`: A slice of vectors representing the first set of input data.
/// * `x1`: A slice of vectors representing the second set of input data.
/// * `kernel`: A function that takes two f64 values and returns a f64 value.
/// * `epsilon`: The kernel function's parameter.
///
/// # Returns
/// A vector of vectors representing the design matrix.
pub fn build_design_matrix<X>(
    x0: &[X],
    x1: &[X],
    kernel: &fn(f64, f64) -> f64,
    epsilon: f64,
) -> Vec<Vec<f64>>
where
    X: Numeric,
{
    (0..x0.len())
        .map(|i| {
            (0..x1.len())
                .map(|j| {
                    let dist = x0[i].squared_distance(&x1[j]).max(f64::EPSILON);
                    kernel(dist, epsilon)
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>()
}

/// Solves a linear system using external BLAS/LAPACK implementations.
///
/// This function is only available when either the "openblas" or "intel-mkl"
/// feature is enabled. It provides better performance for large matrices
/// by leveraging optimized linear algebra libraries.
///
/// # Arguments
/// * `design_matrix` - The coefficient matrix A as a 2D vector of f64 values.
/// * `rhs` - The right-hand side vector with elements of any numeric type T.
///
/// # Returns
/// * `Ok(Vec<T>)` - The solution vector x if successful.
/// * `Err(String)` - An error message if the system cannot be solved.
#[cfg(any(feature = "openblas", feature = "intel-mkl"))]
pub fn ndarray_linear_solver<T: Numeric>(
    design_matrix: &[Vec<f64>],
    rhs: &[T],
) -> Result<Vec<T>, String> {
    let n = design_matrix.len();

    if rhs.len() != n {
        return Err(format!(
            "Design matrix and rhs have different lengths: {} vs {}",
            n,
            rhs.len()
        ));
    }

    // Convert design matrix to Array2
    let design_array =
        Array2::from_shape_vec((n, n), design_matrix.iter().flatten().copied().collect())
            .map_err(|e| format!("Design matrix conversion failed: {}", e))?;

    // Get dimension from first RHS element
    let dim = rhs[0].to_flattened().len();

    // Check all elements have consistent dimension
    for (i, val) in rhs.iter().enumerate() {
        if val.to_flattened().len() != dim {
            return Err(format!(
                "Inconsistent dimension at index {}: expected {}, got {}",
                i,
                dim,
                val.to_flattened().len()
            ));
        }
    }

    // Solve each dimension separately
    let mut weights_matrix = Array2::zeros((n, dim));
    for d in 0..dim {
        // Extract column from RHS
        let rhs_column: Array1<f64> = Array1::from_iter(rhs.iter().map(|x| x.to_flattened()[d]));

        // Solve for this dimension
        let w = design_array
            .solve(&rhs_column)
            .map_err(|e| format!("Linear solve failed for dimension {}: {}", d, e))?;

        weights_matrix.column_mut(d).assign(&w);
    }

    // Convert back to output type
    weights_matrix
        .outer_iter()
        .map(|row| T::from_flattened(row.to_vec()))
        .collect()
}

/// Solves a linear system using faer LU decomposition.
///
/// # Arguments
/// * `design_matrix` - The coefficient matrix A as a 2D vector of f64 values.
/// * `rhs` - The right-hand side vector with elements of any numeric type T.
///
/// # Returns
/// * `Ok(Vec<T>)` - The solution vector x if successful.
/// * `Err(String)` - An error message if the system cannot be solved.
#[cfg(feature = "faer")]
pub fn faer_linear_solver<T: Numeric>(
    design_matrix: &[Vec<f64>],
    rhs: &[T],
) -> Result<Vec<T>, String> {

    let (m, n) = (design_matrix.first().unwrap().len(), design_matrix.len());
    let dim = rhs.first().unwrap().to_flattened().len();

    let dmf: Vec<f64> = design_matrix.iter().flatten().map(|&i| i).collect();
    let a = MatRef::<f64>::from_column_major_slice(&dmf, m, n);
    let mut rhs = Mat::from_fn(m, dim, |a, b| rhs.get(a).unwrap().to_flattened()[b]);

    a.as_ref().partial_piv_lu().solve_adjoint_in_place(rhs.as_mut());
    rhs = rhs.transpose().to_owned();

    let mut w = Vec::with_capacity(dim * n);
    zip!(&rhs).for_each(|unzip!(i)| {
        w.push(*i);
    });

    Ok(w.chunks_exact(dim).map(|chunk| T::from_flattened(chunk.to_vec()).unwrap()).collect())
}

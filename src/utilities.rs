/// Merges the first two pairs of point arrays with the second, if they are unique.
///
/// # Arguments
/// * `x`: The first component array of the first pair.
/// * `y`: The second component array of the first pair.
/// * `x_new`: The first component array of the second pair.
/// * `y_new`: The second component array of the second pair.
pub fn merge_unique_points<X, Y>(x: &mut Vec<X>, y: &mut Vec<Y>, x_new: Vec<X>, y_new: Vec<Y>)
where
    X: PartialEq,
{
    let mut y_iter = y_new.into_iter();
    for point in x_new {
        let y_value = y_iter
            .next()
            .expect("y_new should have the same length as x_new");
        if !x.contains(&point) {
            x.push(point);
            y.push(y_value);
        }
    }
}

fn _remove_duplicates<X, Y>(x: &mut Vec<X>, y: &mut Vec<Y>)
where
    X: PartialEq,
{
    let mut i = 0;
    while i < x.len() {
        let is_duplicate = (0..i).any(|j| x[j] == x[i]);
        if is_duplicate {
            x.remove(i);
            y.remove(i);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_unique_points_1d() {
        let mut x_1d: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut y_1d: Vec<f64> = vec![1.0, 4.0, 9.0];
        let x_new_1d: Vec<f64> = vec![2.0, 3.0, 4.0];
        let y_new_1d: Vec<f64> = vec![4.0, 9.0, 16.0];

        merge_unique_points(&mut x_1d, &mut y_1d, x_new_1d, y_new_1d);

        assert_eq!(x_1d, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(y_1d, vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_merge_unique_points_2d() {
        let mut x_2d: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut y_2d: Vec<f64> = vec![5.0, 6.0];
        let x_new_2d: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        let y_new_2d: Vec<f64> = vec![7.0, 8.0];

        merge_unique_points(&mut x_2d, &mut y_2d, x_new_2d, y_new_2d);

        assert_eq!(x_2d, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(y_2d, vec![5.0, 6.0, 8.0]);
    }

    #[test]
    fn test_merge_unique_points_3d() {
        let mut x_3d: Vec<[f64; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut y_3d: Vec<[f64; 3]> = vec![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let x_new_3d: Vec<[f64; 3]> = vec![[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]];
        let y_new_3d: Vec<[f64; 3]> = vec![[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]];

        merge_unique_points(&mut x_3d, &mut y_3d, x_new_3d, y_new_3d);

        assert_eq!(
            x_3d,
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        );
        assert_eq!(
            y_3d,
            vec![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [16.0, 17.0, 18.0]]
        );
    }
}

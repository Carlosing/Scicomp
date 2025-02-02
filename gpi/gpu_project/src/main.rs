use rayon::prelude::*;

use gpu_project::Matrix;

fn main() {
    let matriz1  =  Matrix::new(2,1,vec![1.1, 2.2]);

    let matriz2= Matrix::new(1,2, vec![2.2,1.1]);

    let mult=matriz1.multiply(&matriz2);

    print!("{}", mult.get(0,0));

    let multpar = matriz1.multiply_parallel(&matriz2);
}




// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_matrix_new() {
//         let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
//         assert_eq!(matrix.rows, 2);
//         assert_eq!(matrix.cols, 2);
//         assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
//     }

//     #[test]
//     #[should_panic(expected = "Incorrect dimensions")]
//     fn test_matrix_new_incorrect_dimensions() {
//         Matrix::new(2, 2, vec![1.0, 2.0, 3.0]);
//     }

//     #[test]
//     fn test_matrix_zeros() {
//         let matrix = Matrix::zeros(2, 2);
//         assert_eq!(matrix.rows, 2);
//         assert_eq!(matrix.cols, 2);
//         assert_eq!(matrix.data, vec![0.0; 4]);
//     }

//     #[test]
//     fn test_matrix_get() {
//         let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
//         assert_eq!(matrix.get(0, 0), 1.0);
//         assert_eq!(matrix.get(1, 1), 4.0);
//     }

//     #[test]
//     fn test_matrix_set() {
//         let mut matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
//         matrix.set(0, 0, 5.0);
//         assert_eq!(matrix.get(0, 0), 5.0);
//     }

//     #[test]
//     fn test_matrix_multiply() {
//         let matrix1 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//         let matrix2 = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
//         let result = matrix1.multiply(&matrix2);

//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.get(0, 0), 58.0);
//         assert_eq!(result.get(0, 1), 64.0);
//         assert_eq!(result.get(1, 0), 139.0);
//         assert_eq!(result.get(1, 1), 154.0);
//     }
// }


// Carlos Alberto Escobedo Lopez

use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};
use std::fmt::Debug;
use num_traits::{Float, Zero};
use std::time::Instant;

fn main() {
    // Parse command-line arguments to get the file name and size value.
    let (file_name, size) = get_args().expect("Error getting args");

    // Construct the full path to the file.
    let file_path = get_path(&file_name);

    // Open the specified file.
    let file = open_file(&file_path).expect("Error opening file");

    // Read the file content into lines, handling potential errors.
    let reader = io::BufReader::new(file).lines();
    let lines: Vec<String> = reader.collect::<Result<_, _>>().expect("Error reading lines");

    // Measure the time for f32 operations
    let start_f32 = Instant::now();

    // Convert the file's lines into a CSR (Compressed Sparse Row) matrix representation.
    let matrixf32: Matrix<f32> = convert_file_csr::<f32>(lines.clone());
    let matrixf32_dense = matrixf32.to_dense();
    let matrix32_conj = matrixf32_dense.transpose_conjugate();

    let Bf32 = matrixf32_dense.matmul(&matrix32_conj);

    let L32 = Bf32.cholesky().expect("Problem with decomposition");
    let L_T32 = L32.transpose_conjugate();

    let b32 = DenseMatrix {
        rows: L32.rows,
        columns: 1,
        data: vec![size.parse::<f32>().unwrap(); L32.rows],
    };

    let y32 = L32.forward_substitution(&b32);
    let x32 = L_T32.backward_substitution(&y32);

    let error_32 = matrixf32_dense.matmul(&x32).subtract(&b32);
    let m_norm_32 = error_32.max_norm(&b32);
    let eu_norm_32 = error_32.euclidean_norm(&b32);

    // Print the results for f32
    let duration_f32 = start_f32.elapsed();
    println!("{}: f32 err_max = {}, err_2 = {}", file_name, m_norm_32, eu_norm_32);
    println!("Time taken for f32: {:?}", duration_f32);

    // Measure the time for f64 operations
    let start_f64 = Instant::now();

    // Convert the file's lines into a CSR (Compressed Sparse Row) matrix representation for f64
    let matrixf64: Matrix<f64> = convert_file_csr::<f64>(lines.clone());
    let matrixf64_dense = matrixf64.to_dense();
    let matrix64_conj = matrixf64_dense.transpose_conjugate();

    let Bf64 = matrixf64_dense.matmul(&matrix64_conj);

    let L64 = Bf64.cholesky().expect("Problem with decomposition");
    let L_T64 = L64.transpose_conjugate();

    let b64 = DenseMatrix {
        rows: L64.rows,
        columns: 1,
        data: vec![size.parse::<f64>().unwrap(); L64.rows],
    };

    let y64 = L64.forward_substitution(&b64);
    let x64 = L_T64.backward_substitution(&y64);

    let error_64 = matrixf64_dense.matmul(&x64).subtract(&b64);
    let m_norm_64 = error_64.max_norm(&b64);
    let eu_norm_64 = error_64.euclidean_norm(&b64);

    // Print the results for f64
    let duration_f64 = start_f64.elapsed();
    println!("{}: f64 err_max = {}, err_2 = {}", file_name, m_norm_64, eu_norm_64);
    println!("Time taken for f64: {:?}", duration_f64);

    
}



struct Matrix<T> {
    rows : usize,
    columns : usize,
    nnz : usize,
    values : Vec<T>,
    col_indices : Vec<usize>,
    row_ptr : Vec<usize>,
}

pub fn get_args() -> Result<(String, String), String> {
    // Function to parse command-line arguments
    let args: Vec<String> = env::args().collect(); // Collecting command-line arguments into a vector
    if args.len() != 3 {
        // Ensuring the correct number of arguments are provided
        return Err("Error: Name or size missing".to_string()); // Returning an error if arguments are missing
    }
    Ok((args[1].clone(), args[2].clone())) // Returning the parsed arguments as a tuple
}

pub fn get_path(filename: &str) -> PathBuf {
    // Function to construct the path to a file
    Path::new("./data").join(filename) // Joining the filename with the "./data" directory
}

pub fn open_file(path: &Path) -> io::Result<File> {
    // Function to open and read the contents of a file
    File::open(path) // Opening the file at the specified path
}


pub trait FloatConvert {
    fn from_f64(val: f64) -> Self;
}

impl FloatConvert for f32 {
    fn from_f64(val: f64) -> Self {
        val as f32
    }
}

impl FloatConvert for f64 {
    fn from_f64(val: f64) -> Self {
        val
    }
}



fn convert_file_csr<T>(lines: Vec<String>) -> Matrix<T>
where
    T: std::str::FromStr + Default + Copy,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptr: Vec<usize>;

    // First, get the dimensions of the matrix
    let sizes: Vec<usize> = lines[0]
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let rows = sizes[0];
    let columns = sizes[1];
    let nnz = sizes[2];

    row_ptr = vec![0; rows + 1];  // row_ptr has a size of rows + 1, as in CSR.

    // Maintain a counter of elements per row
    let mut row_counts = vec![0; rows];

    // Process the data rows
    for line in &lines[1..] {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        
        // Ensure indices are at least 1 to avoid overflow
        let row = tokens[0].parse::<usize>().unwrap();
        let col = tokens[1].parse::<usize>().unwrap();

        if row < 1 || col < 1 {
            panic!("Invalid index: row = {}, column = {}", row, col);
        }

        // Adjust index to 0
        let row = row - 1;  // Adjust row index
        let col = col - 1;  // Adjust column index

        // Check for overflow
        if row >= rows || col >= columns {
            panic!("Index out of range: row = {}, column = {}", row, col);
        }

        let value = tokens[2].parse::<T>().unwrap();

        values.push(value);
        col_indices.push(col);
        row_counts[row] += 1;
    }

    // Now build row_ptr by accumulating the number of elements per row.
    for i in 1..=rows {
        row_ptr[i] = row_ptr[i - 1] + row_counts[i - 1];
    }

    // Final check for row_ptr
    if row_ptr[rows] != nnz {
        panic!("The total number of elements does not match nnz. row_ptr[{}] = {}", rows, row_ptr[rows]);
    }

    Matrix {
        rows,
        columns,
        nnz,
        values,
        col_indices,
        row_ptr,
    }
}




struct DenseMatrix<T> {
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl<T> DenseMatrix<T>
where
    T: Clone + Default,
{
    /// Creates a new dense matrix with default values.
    pub fn new(rows: usize, columns: usize) -> Self {
        let data = vec![T::default(); rows * columns];
        DenseMatrix { rows, columns, data }
    }

    /// Sets a value at the (row, column) position.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.columns, "Indices out of range");
        self.data[row * self.columns + col] = value;
    }

    /// Gets a value at the (row, column) position.
    pub fn get(&self, row: usize, col: usize) -> &T {
        assert!(row < self.rows && col < self.columns, "Indices out of range");
        &self.data[row * self.columns + col]
    }
}



/// Converts a sparse matrix to a dense matrix representation.
///
/// This method iterates over the non-zero elements of the sparse matrix
/// and sets the corresponding values in a newly created dense matrix.
///
/// # Returns
/// 
/// A `DenseMatrix<T>` containing the same elements as the sparse matrix.
///
/// # Example
///
/// ```
/// let sparse_matrix = SparseMatrix::new...;
/// let dense_matrix = sparse_matrix.to_dense();
/// ```
impl<T> Matrix<T>
where
    T: Clone + Default + Copy,
    
{
    pub fn to_dense(&self) -> DenseMatrix<T>
     {
        let mut dense = DenseMatrix::new(self.rows, self.columns);

        for row in 0..self.rows {
            for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_indices[idx];  // Assuming zero-based indexing
                let value = self.values[idx];
                dense.set(row, col, value);
            }
        }

        dense
    }
}





impl<T> DenseMatrix<T>
where
T: Float + Clone + Default,
{


    pub fn subtract(&self, other: &DenseMatrix<T>) -> DenseMatrix<T> {
    /// Subtracts another `DenseMatrix` from `self` and returns the result as a new `DenseMatrix`.
    /// Panics if the dimensions of the two matrices do not match.
        assert_eq!(self.rows, other.rows, "Matrices must have the same number of rows");
        assert_eq!(self.columns, other.columns, "Matrices must have the same number of columns");

        let mut result = DenseMatrix::new(self.rows, self.columns);

        for i in 0..self.rows {
            for j in 0..self.columns {
                let value = *self.get(i, j) - *other.get(i, j);
                result.set(i, j, value);
            }
        }

        result
    }



    /// Performs Cholesky decomposition for the dense matrix.
    /// Returns a new lower triangular matrix `L` such that A = L * Lᵀ.
    pub fn cholesky(&self) -> Result<DenseMatrix<T>, String> {
        if self.rows != self.columns {
            return Err("The matrix must be square to perform Cholesky decomposition".to_string());
        }

        let n = self.rows;
        let mut l = DenseMatrix::new(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = T::zero();

                for k in 0..j {
                    sum = sum + *l.get(i, k) * *l.get(j, k);
                }

                if i == j {
                    let value = *self.get(i, i) - sum;
                    if value <= T::zero() {
                        return Err(format!(
                            "The matrix is not positive definite. Found a negative or zero value on the diagonal at position ({}, {})",
                            i, i
                        ));
                    }
                    l.set(i, j, value.sqrt());
                } else {
                    let value = (*self.get(i, j) - sum) / *l.get(j, j);
                    l.set(i, j, value);
                }
            }
        }

        Ok(l)
    }
}



impl<T> DenseMatrix<T>
where
    T: Float + Clone + Default,
{
    /// Returns the transpose of the matrix (conjugate for complex numbers).
    pub fn transpose_conjugate(&self) -> DenseMatrix<T> {
        // Creamos una nueva matriz para la transposición
        let mut transposed = DenseMatrix::new(self.columns, self.rows); // Invertimos filas y columnas
        
        // Recorremos la matriz original y asignamos los valores en la transpuesta
        for row in 0..self.rows {
            for col in 0..self.columns {
                // Obtenemos el valor de la matriz original
                let value = self.get(row, col);
                
                // Colocamos el valor en la posición transpuesta
                transposed.set(col, row, *value); // Intercambiamos las posiciones
            }
        }
        
        transposed
    }
}



impl<T> DenseMatrix<T>
where
T: Float + Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Performs dense * dense matrix multiplication.
    pub fn matmul(&self, other: &DenseMatrix<T>) -> DenseMatrix<T> {
        assert_eq!(
            self.columns, other.rows,
            "The number of columns of the first matrix must match the number of rows of the second matrix"
        );

        // Create a new matrix to store the result.
        let mut result = DenseMatrix::new(self.rows, other.columns);

        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum = T::zero();
                for k in 0..self.columns {
                    sum = sum + (*self.get(i, k) * *other.get(k, j));
                }
                result.set(i, j, sum);
            }
        }

        result
    }
}


impl<T> DenseMatrix<T>
where
    T: Float + Clone + Default + Debug,
{
    /// Performs forward substitution to solve Lx = b.
    /// The matrix L must be lower triangular.
    pub fn forward_substitution(&self, b: &DenseMatrix<T>) -> DenseMatrix<T> {
        assert_eq!(self.rows, self.columns, "The matrix must be square");
        assert_eq!(b.columns, 1, "The input b must be a column vector");
        assert_eq!(self.rows, b.rows, "The number of rows in L must match the size of b");

        let mut x = vec![T::zero(); self.rows];

        for i in 0..self.rows {
            let mut sum = T::zero();
            for j in 0..i {
                sum = sum + *self.get(i, j) * x[j];
            }
            let value = *b.get(i, 0) - sum;
            let diag = self.get(i, i);
            assert_ne!(*diag, T::zero(), "The diagonal element cannot be zero");
            x[i] = value / *diag;
        }

        DenseMatrix {
            rows: x.len(),
            columns: 1,
            data: x,
        }
    }
}





impl<T> DenseMatrix<T>
where
    T: Float + Clone + Default + Debug + std::ops::AddAssign,
{
    /// Performs backward substitution to solve Ux = b.
    /// The matrix U must be upper triangular.
    pub fn backward_substitution(&self, b: &DenseMatrix<T>) -> DenseMatrix<T> {
        assert_eq!(self.rows, self.columns, "The matrix must be square");
        assert_eq!(self.rows, b.rows, "The length of b must match the rows of the matrix");

        let mut x = vec![T::zero(); self.rows];

        for i in (0..self.rows).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..self.columns {
                sum += *self.get(i, j) * x[j];
            }
            let value = *b.get(i, 0) - sum;
            let diag = *self.get(i, i);
            assert_ne!(diag,T::zero(), "The diagonal element cannot be zero");
            x[i] = value / diag;
        }

        DenseMatrix {
            rows: x.len(),
            columns: 1,
            data: x,
        }
    }
}






impl<T> DenseMatrix<T>
where
    T: Float + Zero + PartialOrd + std::iter::Sum,
{
    /// Calculates the maximum norm between two vectors.
    pub fn max_norm(&self, other: &DenseMatrix<T>) -> T {
        assert_eq!(self.rows, other.rows, "Matrices must have the same number of rows");
        assert_eq!(self.columns, other.columns, "Matrices must have the same number of columns");

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (*a - *b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(T::zero())
    }

    /// Calculates the Euclidean norm between two vectors.
    pub fn euclidean_norm(&self, other: &DenseMatrix<T>) -> T {
        assert_eq!(self.rows, other.rows, "Matrices must have the same number of rows");
        assert_eq!(self.columns, other.columns, "Matrices must have the same number of columns");

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (*a - *b).powi(2))
            .sum::<T>()
            .sqrt()
    }
}




// Tests with `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_file_csr() {
        let input = vec![
            String::from("4 4 4"),
            String::from("1 1 1.0"),
            String::from("2 2 2.0"),
            String::from("3 3 3.0"),
            String::from("4 4 4.0"),
        ];

        let matrix: Matrix<f32> = convert_file_csr(input);

        assert_eq!(matrix.rows, 4);
        assert_eq!(matrix.columns, 4);
        assert_eq!(matrix.nnz, 4);
        assert_eq!(matrix.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.col_indices, vec![0, 1, 2, 3]);
        assert_eq!(matrix.row_ptr, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_new() {
        let matrix: DenseMatrix<f32> = DenseMatrix::new(3, 3);

        // Verify the dimensions of the matrix
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.columns, 3);

        // Verify that all initial values are 0.0 (default value for f32)
        for i in 0..matrix.rows {
            for j in 0..matrix.columns {
                assert_eq!(*matrix.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_set() {
        let mut matrix: DenseMatrix<f32> = DenseMatrix::new(3, 3);

        // Set some values
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 3.0);

        // Verify that the values were correctly assigned
        assert_eq!(*matrix.get(0, 0), 1.0);
        assert_eq!(*matrix.get(1, 1), 2.0);
        assert_eq!(*matrix.get(2, 2), 3.0);

        // Verify that the other positions are still 0.0
        assert_eq!(*matrix.get(0, 1), 0.0);
        assert_eq!(*matrix.get(1, 0), 0.0);
        assert_eq!(*matrix.get(2, 1), 0.0);
    }

    #[test]
    fn test_get() {
        let mut matrix: DenseMatrix<f32> = DenseMatrix::new(3, 3);

        // Set some values
        matrix.set(0, 0, 5.0);
        matrix.set(1, 1, 10.0);
        matrix.set(2, 2, 15.0);

        // Verify that the value of the specific positions is correct
        assert_eq!(*matrix.get(0, 0), 5.0);
        assert_eq!(*matrix.get(1, 1), 10.0);
        assert_eq!(*matrix.get(2, 2), 15.0);

        // Verify that getting an unset position gives the default value (0.0)
        assert_eq!(*matrix.get(0, 1), 0.0);
    }

    #[test]
    fn test_to_dense() {
        // Create a sparse matrix manually (Sparse Matrix structure)
        let sparse_matrix = Matrix {
            rows: 3,
            columns: 3,
            nnz: 3,  // Number of non-zero elements
            values: vec![1.0, 2.0, 3.0],  // Non-zero values
            col_indices: vec![0, 1, 2],  // Column indices
            row_ptr: vec![0, 1, 2, 3],  // Row pointers (last element is the number of nnz)
        };

        // Convert the sparse matrix to a dense matrix
        let dense_matrix = sparse_matrix.to_dense();

        // Verify the dimensions
        assert_eq!(dense_matrix.rows, 3);
        assert_eq!(dense_matrix.columns, 3);

        // Verify that the values were correctly copied in the dense matrix
        assert_eq!(*dense_matrix.get(0, 0), 1.0);  // First non-zero value
        assert_eq!(*dense_matrix.get(1, 1), 2.0);  // Second non-zero value
        assert_eq!(*dense_matrix.get(2, 2), 3.0);  // Third non-zero value

        // Verify that the other positions are still 0.0
        assert_eq!(*dense_matrix.get(0, 1), 0.0);  // Implicit zero
        assert_eq!(*dense_matrix.get(0, 2), 0.0);  // Implicit zero
        assert_eq!(*dense_matrix.get(1, 0), 0.0);  // Implicit zero
        assert_eq!(*dense_matrix.get(2, 0), 0.0);  // Implicit zero
        assert_eq!(*dense_matrix.get(2, 1), 0.0);  // Implicit zero
    }

    #[test]
    fn test_set_2x3() {
        let mut matrix = DenseMatrix::new(2, 3);
        
        // Set values in the matrix
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(0, 2, 3.0);
        matrix.set(1, 0, 4.0);
        matrix.set(1, 1, 5.0);
        matrix.set(1, 2, 6.0);
        
        // Verify that the values are correctly set
        assert_eq!(*matrix.get(0, 0), 1.0);  // Row 0, Column 0
        assert_eq!(*matrix.get(0, 1), 2.0);  // Row 0, Column 1
        assert_eq!(*matrix.get(0, 2), 3.0);  // Row 0, Column 2
        assert_eq!(*matrix.get(1, 0), 4.0);  // Row 1, Column 0
        assert_eq!(*matrix.get(1, 1), 5.0);  // Row 1, Column 1
        assert_eq!(*matrix.get(1, 2), 6.0);  // Row 1, Column 2
    }

    #[test]
    fn test_set_and_get_1x1() {
        let mut matrix = DenseMatrix::new(1, 1);
        matrix.set(0, 0, 10.0);
        
        // Verify that the set value is correct
        assert_eq!(*matrix.get(0, 0), 10.0);
    }

    #[test]
    fn test_set_and_get_2x2() {
        let mut matrix = DenseMatrix::new(2, 2);
        
        // Set values in various positions
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        // Verify that the values are correctly set
        assert_eq!(*matrix.get(0, 0), 1.0);
        assert_eq!(*matrix.get(0, 1), 2.0);
        assert_eq!(*matrix.get(1, 0), 3.0);
        assert_eq!(*matrix.get(1, 1), 4.0);
    }

    #[test]
    fn test_set_and_get_3x3() {
        let mut matrix = DenseMatrix::new(3, 3);
        
        // Set values in various positions
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(0, 2, 3.0);
        matrix.set(1, 0, 4.0);
        matrix.set(1, 1, 5.0);
        matrix.set(1, 2, 6.0);
        matrix.set(2, 0, 7.0);
        matrix.set(2, 1, 8.0);
        matrix.set(2, 2, 9.0);

        // Verify all the set values
        assert_eq!(*matrix.get(0, 0), 1.0);
        assert_eq!(*matrix.get(0, 1), 2.0);
        assert_eq!(*matrix.get(0, 2), 3.0);
        assert_eq!(*matrix.get(1, 0), 4.0);
        assert_eq!(*matrix.get(1, 1), 5.0);
        assert_eq!(*matrix.get(1, 2), 6.0);
        assert_eq!(*matrix.get(2, 0), 7.0);
        assert_eq!(*matrix.get(2, 1), 8.0);
        assert_eq!(*matrix.get(2, 2), 9.0);
    }

    #[test]
    fn test_set_and_get_edges() {
        let mut matrix = DenseMatrix::new(3, 3);
        
        // Set values in the corners of the matrix
        matrix.set(0, 0, 1.0); // top left corner
        matrix.set(0, 2, 3.0); // top right corner
        matrix.set(2, 0, 7.0); // bottom left corner
        matrix.set(2, 2, 9.0); // bottom right corner

        // Verify that the values set in the corners are correct
        assert_eq!(*matrix.get(0, 0), 1.0);
        assert_eq!(*matrix.get(0, 2), 3.0);
        assert_eq!(*matrix.get(2, 0), 7.0);
        assert_eq!(*matrix.get(2, 2), 9.0);
    }

    #[test]
    fn test_transpose_2x2() {
        let mut matrix = DenseMatrix::new(2, 2);
        
        // Set values in the original matrix
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);
        
        // The transpose of the matrix should swap rows and columns
        let transposed = matrix.transpose_conjugate();
        
        // Verify that the values are correct
        assert_eq!(*transposed.get(0, 0), 1.0);
        assert_eq!(*transposed.get(0, 1), 3.0);
        assert_eq!(*transposed.get(1, 0), 2.0);
        assert_eq!(*transposed.get(1, 1), 4.0);
    }

    #[test]
    fn test_transpose_conjugate_3x2() {
        let mut matrix = DenseMatrix::new(3, 2);
        
        // Set values in the original matrix
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);
        matrix.set(2, 0, 5.0);
        matrix.set(2, 1, 6.0);
        
        // The transpose of the matrix should swap rows and columns
        let transposed = matrix.transpose_conjugate();
        
        // Verify that the values are correct
        assert_eq!(*transposed.get(0, 0), 1.0);
        assert_eq!(*transposed.get(1, 0), 2.0);
        assert_eq!(*transposed.get(0, 1), 3.0);
        assert_eq!(*transposed.get(1, 1), 4.0);
        assert_eq!(*transposed.get(0, 2), 5.0);
        assert_eq!(*transposed.get(1, 2), 6.0);
    }

    #[test]
    fn test_get_2x3() {
        let mut matrix = DenseMatrix::new(2, 3);
        
        // Set values in the matrix
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(0, 2, 3.0);
        matrix.set(1, 0, 4.0);
        matrix.set(1, 1, 5.0);
        matrix.set(1, 2, 6.0);
        
        // Verify that the values obtained are correct
        assert_eq!(*matrix.get(0, 0), 1.0);  // Row 0, Column 0
        assert_eq!(*matrix.get(0, 1), 2.0);  // Row 0, Column 1
        assert_eq!(*matrix.get(0, 2), 3.0);  // Row 0, Column 2
        assert_eq!(*matrix.get(1, 0), 4.0);  // Row 1, Column 0
        assert_eq!(*matrix.get(1, 1), 5.0);  // Row 1, Column 1
        assert_eq!(*matrix.get(1, 2), 6.0);  // Row 1, Column 2
    }

    #[test]
    fn test_transpose_conjugate_2x3() {
        let mut matrix = DenseMatrix::new(2, 3);
        
        // Set values in the original matrix
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(0, 2, 3.0);
        matrix.set(1, 0, 4.0);
        matrix.set(1, 1, 5.0);
        matrix.set(1, 2, 6.0);
        
        // The transpose of the matrix should swap rows and columns
        let transposed = matrix.transpose_conjugate();
        
        println!("{:?}", matrix.data);
        println!("{:?}", transposed.data);

        // Verify that the values are correct
        assert_eq!(*transposed.get(0, 0), 1.0);
        assert_eq!(*transposed.get(0, 1), 4.0);
        assert_eq!(*transposed.get(1, 0), 2.0);
        assert_eq!(*transposed.get(1, 1), 5.0);
        assert_eq!(*transposed.get(2, 0), 3.0);
        assert_eq!(*transposed.get(2, 1), 6.0);
    }

    fn test_transpose_manual() {
        // Step 1: Create an original 2x3 matrix
        let mut original = DenseMatrix::new(2, 3); 
        original.set(0, 0, 1.0);
        original.set(0, 1, 2.0);
        original.set(0, 2, 3.0);
        original.set(1, 0, 4.0);
        original.set(1, 1, 5.0);
        original.set(1, 2, 6.0);

        // Step 2: Create the transposed 3x2 matrix
        let mut transposed = DenseMatrix::new(3, 2);

        // Step 3: Use get and set to manually transpose the matrix
        transposed.set(0, 0, *original.get(0, 0)); // (0, 0) -> (0, 0)
        transposed.set(1, 0, *original.get(1, 0)); // (1, 0) -> (0, 1)
        transposed.set(0, 1, *original.get(0, 1)); // (0, 1) -> (1, 0)
        transposed.set(1, 1, *original.get(1, 1)); // (1, 1) -> (1, 1)
        transposed.set(2, 0, *original.get(0, 2)); // (0, 2) -> (2, 0)
        transposed.set(2, 1, *original.get(1, 2)); // (1, 2) -> (2, 1)

        // Step 4: Verify that the transposition is correct
        assert_eq!(*transposed.get(0, 0), 1.0); // First value of the transposed matrix
        assert_eq!(*transposed.get(1, 0), 2.0); // Second value of the transposed matrix
        assert_eq!(*transposed.get(2, 0), 3.0); // Third value of the transposed matrix

        assert_eq!(*transposed.get(0, 1), 4.0); // First value of the second column transposed
        assert_eq!(*transposed.get(1, 1), 5.0); // Second value of the second column transposed
        assert_eq!(*transposed.get(2, 1), 6.0); // Third value of the second column transposed
    }

    #[test]
fn test_matmul_2x2() {
    let mut matrix_a = DenseMatrix::new(2, 2);
    matrix_a.set(0, 0, 1.0);
    matrix_a.set(0, 1, 2.0);
    matrix_a.set(1, 0, 3.0);
    matrix_a.set(1, 1, 4.0);

    let mut matrix_b = DenseMatrix::new(2, 2);
    matrix_b.set(0, 0, 2.0);
    matrix_b.set(0, 1, 0.0);
    matrix_b.set(1, 0, 1.0);
    matrix_b.set(1, 1, 2.0);

    let result = matrix_a.matmul(&matrix_b);

    assert_eq!(*result.get(0, 0), 4.0);
    assert_eq!(*result.get(0, 1), 4.0);
    assert_eq!(*result.get(1, 0), 10.0);
    assert_eq!(*result.get(1, 1), 8.0);
}

#[test]
fn test_matmul_3x2_2x3() {
    let mut matrix_a = DenseMatrix::new(3, 2);
    matrix_a.set(0, 0, 1.0);
    matrix_a.set(0, 1, 2.0);
    matrix_a.set(1, 0, 3.0);
    matrix_a.set(1, 1, 4.0);
    matrix_a.set(2, 0, 5.0);
    matrix_a.set(2, 1, 6.0);

    let mut matrix_b = DenseMatrix::new(2, 3);
    matrix_b.set(0, 0, 7.0);
    matrix_b.set(0, 1, 8.0);
    matrix_b.set(0, 2, 9.0);
    matrix_b.set(1, 0, 10.0);
    matrix_b.set(1, 1, 11.0);
    matrix_b.set(1, 2, 12.0);

    let result = matrix_a.matmul(&matrix_b);

    assert_eq!(*result.get(0, 0), 27.0);
    assert_eq!(*result.get(0, 1), 30.0);
    assert_eq!(*result.get(0, 2), 33.0);
    assert_eq!(*result.get(1, 0), 61.0);
    assert_eq!(*result.get(1, 1), 68.0);
    assert_eq!(*result.get(1, 2), 75.0);
    assert_eq!(*result.get(2, 0), 95.0);
    assert_eq!(*result.get(2, 1), 106.0);
    assert_eq!(*result.get(2, 2), 117.0);
}

#[test]
fn test_matmul_1x3_3x1() {
    let mut matrix_a = DenseMatrix::new(1, 3);
    matrix_a.set(0, 0, 1.0);
    matrix_a.set(0, 1, 2.0);
    matrix_a.set(0, 2, 3.0);

    let mut matrix_b = DenseMatrix::new(3, 1);
    matrix_b.set(0, 0, 4.0);
    matrix_b.set(1, 0, 5.0);
    matrix_b.set(2, 0, 6.0);

    let result = matrix_a.matmul(&matrix_b);

    assert_eq!(*result.get(0, 0), 32.0);
}

use super::*;

#[test]
fn test_subtract_2x2() {
    let mut matrix_a = DenseMatrix::new(2, 2);
    matrix_a.set(0, 0, 5.0);
    matrix_a.set(0, 1, 6.0);
    matrix_a.set(1, 0, 7.0);
    matrix_a.set(1, 1, 8.0);

    let mut matrix_b = DenseMatrix::new(2, 2);
    matrix_b.set(0, 0, 1.0);
    matrix_b.set(0, 1, 2.0);
    matrix_b.set(1, 0, 3.0);
    matrix_b.set(1, 1, 4.0);

    let result = matrix_a.subtract(&matrix_b);

    assert_eq!(*result.get(0, 0), 4.0);
    assert_eq!(*result.get(0, 1), 4.0);
    assert_eq!(*result.get(1, 0), 4.0);
    assert_eq!(*result.get(1, 1), 4.0);
}

#[test]
fn test_subtract_3x3() {
    let mut matrix_a = DenseMatrix::new(3, 3);
    matrix_a.set(0, 0, 9.0);
    matrix_a.set(0, 1, 8.0);
    matrix_a.set(0, 2, 7.0);
    matrix_a.set(1, 0, 6.0);
    matrix_a.set(1, 1, 5.0);
    matrix_a.set(1, 2, 4.0);
    matrix_a.set(2, 0, 3.0);
    matrix_a.set(2, 1, 2.0);
    matrix_a.set(2, 2, 1.0);

    let mut matrix_b = DenseMatrix::new(3, 3);
    matrix_b.set(0, 0, 1.0);
    matrix_b.set(0, 1, 2.0);
    matrix_b.set(0, 2, 3.0);
    matrix_b.set(1, 0, 4.0);
    matrix_b.set(1, 1, 5.0);
    matrix_b.set(1, 2, 6.0);
    matrix_b.set(2, 0, 7.0);
    matrix_b.set(2, 1, 8.0);
    matrix_b.set(2, 2, 9.0);

    let result = matrix_a.subtract(&matrix_b);

    assert_eq!(*result.get(0, 0), 8.0);
    assert_eq!(*result.get(0, 1), 6.0);
    assert_eq!(*result.get(0, 2), 4.0);
    assert_eq!(*result.get(1, 0), 2.0);
    assert_eq!(*result.get(1, 1), 0.0);
    assert_eq!(*result.get(1, 2), -2.0);
    assert_eq!(*result.get(2, 0), -4.0);
    assert_eq!(*result.get(2, 1), -6.0);
    assert_eq!(*result.get(2, 2), -8.0);
}

#[test]
fn test_subtract_1x1() {
    let mut matrix_a = DenseMatrix::new(1, 1);
    matrix_a.set(0, 0, 10.0);

    let mut matrix_b = DenseMatrix::new(1, 1);
    matrix_b.set(0, 0, 5.0);

    let result = matrix_a.subtract(&matrix_b);

    assert_eq!(*result.get(0, 0), 5.0);
}

#[test]
fn test_subtract_2x3() {
    let mut matrix_a = DenseMatrix::new(2, 3);
    matrix_a.set(0, 0, 1.0);
    matrix_a.set(0, 1, 2.0);
    matrix_a.set(0, 2, 3.0);
    matrix_a.set(1, 0, 4.0);
    matrix_a.set(1, 1, 5.0);
    matrix_a.set(1, 2, 6.0);

    let mut matrix_b = DenseMatrix::new(2, 3);
    matrix_b.set(0, 0, 6.0);
    matrix_b.set(0, 1, 5.0);
    matrix_b.set(0, 2, 4.0);
    matrix_b.set(1, 0, 3.0);
    matrix_b.set(1, 1, 2.0);
    matrix_b.set(1, 2, 1.0);

    let result = matrix_a.subtract(&matrix_b);

    assert_eq!(*result.get(0, 0), -5.0);
    assert_eq!(*result.get(0, 1), -3.0);
    assert_eq!(*result.get(0, 2), -1.0);
    assert_eq!(*result.get(1, 0), 1.0);
    assert_eq!(*result.get(1, 1), 3.0);
    assert_eq!(*result.get(1, 2), 5.0);
}

#[test]
fn test_cholesky_2x2() {
    let mut matrix = DenseMatrix::new(2, 2);
    matrix.set(0, 0, 4.0);
    matrix.set(0, 1, 12.0);
    matrix.set(1, 0, 12.0);
    matrix.set(1, 1, 37.0);

    let l = matrix.cholesky().expect("Cholesky decomposition failed");

    let tolerance = 1e-6;
    assert!((*l.get(0, 0) - 2.0).abs() < tolerance);
    assert!((*l.get(0, 1) - 0.0).abs() < tolerance);
    assert!((*l.get(1, 0) - 6.0).abs() < tolerance);
    assert!((*l.get(1, 1) - 1.0).abs() < tolerance);
}

#[test]
fn test_cholesky_3x3() {
    let mut matrix = DenseMatrix::new(3, 3);
    matrix.set(0, 0, 25.0);
    matrix.set(0, 1, 15.0);
    matrix.set(0, 2, -5.0);
    matrix.set(1, 0, 15.0);
    matrix.set(1, 1, 18.0);
    matrix.set(1, 2, 0.0);
    matrix.set(2, 0, -5.0);
    matrix.set(2, 1, 0.0);
    matrix.set(2, 2, 11.0);

    let l = matrix.cholesky().expect("Cholesky decomposition failed");

    let tolerance = 1e-6;
    assert!((*l.get(0, 0) - 5.0).abs() < tolerance);
    assert!((*l.get(0, 1) - 0.0).abs() < tolerance);
    assert!((*l.get(0, 2) - 0.0).abs() < tolerance);
    assert!((*l.get(1, 0) - 3.0).abs() < tolerance);
    assert!((*l.get(1, 1) - 3.0).abs() < tolerance);
    assert!((*l.get(1, 2) - 0.0).abs() < tolerance);
    assert!((*l.get(2, 0) - -1.0).abs() < tolerance);
    assert!((*l.get(2, 1) - 1.0).abs() < tolerance);
    assert!((*l.get(2, 2) - 3.0).abs() < tolerance);
}

#[test]
fn test_cholesky_4x4() {
    let mut matrix = DenseMatrix::new(4, 4);
    matrix.set(0, 0, 16.0);
    matrix.set(0, 1, 24.0);
    matrix.set(0, 2, 8.0);
    matrix.set(0, 3, 4.0);
    matrix.set(1, 0, 24.0);
    matrix.set(1, 1, 61.0);
    matrix.set(1, 2, 25.0);
    matrix.set(1, 3, 10.0);
    matrix.set(2, 0, 8.0);
    matrix.set(2, 1, 25.0);
    matrix.set(2, 2, 26.0);
    matrix.set(2, 3, 5.0);
    matrix.set(3, 0, 4.0);
    matrix.set(3, 1, 10.0);
    matrix.set(3, 2, 5.0);
    matrix.set(3, 3, 6.0);

    let l = matrix.cholesky().expect("Cholesky decomposition failed");

    let tolerance = 1e1;
    assert!((*l.get(0, 0) - 4.0).abs() < tolerance);
    assert!((*l.get(0, 1) - 0.0).abs() < tolerance);
    assert!((*l.get(0, 2) - 0.0).abs() < tolerance);
    assert!((*l.get(0, 3) - 0.0).abs() < tolerance);
    assert!((*l.get(1, 0) - 6.0).abs() < tolerance);
    assert!((*l.get(1, 1) - 5.0).abs() < tolerance);
    assert!((*l.get(1, 2) - 0.0).abs() < tolerance);
    assert!((*l.get(1, 3) - 0.0).abs() < tolerance);
    assert!((*l.get(2, 0) - 2.0).abs() < tolerance);
    assert!((*l.get(2, 1) - 3.0).abs() < tolerance);
    assert!((*l.get(2, 2) - 3.0).abs() < tolerance);
    assert!((*l.get(2, 3) - 0.0).abs() < tolerance);
    assert!((*l.get(3, 0) - 1.0).abs() < tolerance);
    assert!((*l.get(3, 1) - 1.0).abs() < tolerance);
    assert!((*l.get(3, 2) - 1.0).abs() < tolerance);
    assert!((*l.get(3, 3) - 2.0).abs() < tolerance);
}

#[test]
fn test_non_positive_definite_matrix() {
    // Non-positive definite matrix (has a negative value on the diagonal)
    let matrix_data = vec![
        1.0, 2.0, 3.0,
        2.0, -1.0, 4.0,
        3.0, 4.0, 1.0,
    ];

    let matrix = DenseMatrix {
        rows: 3,
        columns: 3,
        data: matrix_data,
    };

    // Attempt Cholesky decomposition
    let result = matrix.cholesky();
    
    // Verify that an error is returned because the matrix is not positive definite
    assert!(result.is_err(), "Expected an error due to non-positive definite matrix");
    if let Err(err) = result {
        assert!(err.contains("The matrix is not positive definite"), "Error message should mention positive definite condition");
    }
}

#[test]
fn test_forward_substitution() {
    // Create a 3x3 lower triangular matrix with f64
    let mut l = DenseMatrix::<f64>::new(3, 3);
    l.set(0, 0, 2.0);
    l.set(1, 0, 3.0);
    l.set(1, 1, 4.0);
    l.set(2, 0, 1.0);
    l.set(2, 1, 2.0);
    l.set(2, 2, 5.0);

    // Create the vector b with f64
    let mut b = DenseMatrix::<f64>::new(3, 1);
    b.set(0, 0, 4.0);
    b.set(1, 0, 10.0);
    b.set(2, 0, 7.0);

    // Perform forward substitution
    let x = l.forward_substitution(&b);

    // Verify the results with tolerance for floating-point comparison
    let tolerance = 1e0;
    assert!((*x.get(0, 0) - 2.0).abs() < tolerance);
    assert!((*x.get(1, 0) - 1.0).abs() < tolerance);
    assert!((*x.get(2, 0) - 1.0).abs() < tolerance);
}

#[test]
fn test_backward_substitution() {
    // Create a 3x3 upper triangular matrix with f64
    let mut u = DenseMatrix::<f64>::new(3, 3);
    u.set(0, 0, 2.0);
    u.set(0, 1, 3.0);
    u.set(0, 2, 1.0);
    u.set(1, 1, 4.0);
    u.set(1, 2, 2.0);
    u.set(2, 2, 5.0);

    // Create the vector b with f64
    let mut b = DenseMatrix::<f64>::new(3, 1);
    b.set(0, 0, 5.0);
    b.set(1, 0, 6.0);
    b.set(2, 0, 7.0);

    // Perform backward substitution
    let x = u.backward_substitution(&b);

    // Verify the results with tolerance for floating-point comparison
    let tolerance = 1e1;
    assert!((*x.get(0, 0) - (-1.0)).abs() < tolerance);
    assert!((*x.get(1, 0) - 1.0).abs() < tolerance);
    assert!((*x.get(2, 0) - 1.4).abs() < tolerance);
}

}

// Carlos Alberto Escobedo Lopez

use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};
use std::fmt::Debug;

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

    // Convert the file's lines into a CSR (Compressed Sparse Row) matrix representation.
    let mut matrix = convert_file_csr(lines);

    // Convert the CSR matrix to a dense matrix for further processing.
    let matrix = matrix.to_dense();

    // Compute the conjugate transpose of the dense matrix.
    let matrix_conj = matrix.transpose_conjugate();

    // Perform matrix multiplication to obtain B = A * Aᵀ (A-transpose).
    let B = matrix.matmul(&matrix_conj);

    // Perform Cholesky decomposition to obtain a lower triangular matrix L such that B = L * Lᵀ.
    let L = B.cholesky().expect("Problem with decomposition");

    // Compute the transpose conjugate of the lower triangular matrix.
    let L_T = L.transpose_conjugate();

    // Validate the decomposition by recomputing the product L * Lᵀ.
    let decomp_prod = L.matmul(&L_T);

    // Prepare a vector `b` with the specified size as its values.
    let b = DenseMatrix {
        rows: decomp_prod.rows,
        columns: 1,
        data: vec![size.parse::<f64>().unwrap(); decomp_prod.rows],
    };

    // Solve the system Lx = b using forward substitution to obtain `x`.
    let x = decomp_prod.forward_substitution(&b);

    // Calculate the error vector as B * x.
    let error = B.matmul(&x);

    // Compute error metrics: maximum norm and Euclidean norm.
    let m_norm = error.max_norm(&b);
    let eu_norm = error.euclidean_norm(&b);

    // Output the results, including the error norms.
    println!("{}: err_max = {}, err_2 = {}", file_name, m_norm, eu_norm);
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

pub fn convert_file_csr(lines: Vec<String>) -> Matrix<f64> {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();

    let line1 = &lines[0];
    let sizes: Vec<usize> = line1.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let rows = sizes[0];
    let columns = sizes[1];
    let nnz = sizes[2];

    let mut row_ptr = vec![0; rows + 1];
    let mut row_count = vec![0; rows];

    for line in lines.iter().skip(1) {
        let entries: Vec<&str> = line.split_whitespace().collect();
        values.push(entries[2].parse().expect("Error casting"));
        let row: usize = entries[0].parse().expect("Error casting");
        col_indices.push(entries[1].parse().expect("Error casting"));
        row_count[row - 1] += 1;
    }

    for i in 1..=rows {
        row_ptr[i] = row_count[i - 1] + row_ptr[i - 1];
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

impl Matrix<f64> {
    pub fn to_dense(&self) -> DenseMatrix<f64> {
        let mut dense = DenseMatrix::new(self.rows, self.columns);

        for row in 0..self.rows {
            for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_indices[idx] - 1;
                let value = self.values[idx];
                dense.set(row, col, value);
            }
        }

        dense
    }
}

impl DenseMatrix<f64> {
    /// Performs Cholesky decomposition for the dense matrix.
    /// Returns a new lower triangular matrix `L` such that A = L * Lᵀ.
    pub fn cholesky(&self) -> Result<DenseMatrix<f64>, String> {
        if self.rows != self.columns {
            return Err("The matrix must be square to perform Cholesky decomposition".to_string());
        }

        let n = self.rows;
        let mut l = DenseMatrix::new(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }

                if i == j {
                    let value = self.get(i, i) - sum;
                    if value <= 0.0 {
                        return Err(format!(
                            "The matrix is not positive definite. Found a negative or zero value on the diagonal at position ({}, {})",
                            i, i
                        ));
                    }
                    l.set(i, j, value.sqrt());
                } else {
                    let value = (self.get(i, j) - sum) / l.get(j, j);
                    l.set(i, j, value);
                }
            }
        }

        Ok(l)
    }
}

impl DenseMatrix<f64> {
    /// Returns the transpose of the matrix (conjugate for complex numbers).
    pub fn transpose_conjugate(&self) -> DenseMatrix<f64> {
        let mut transposed = DenseMatrix::new(self.columns, self.rows); // Swap rows and columns
        for row in 0..self.rows {
            for col in 0..self.columns {
                let value = self.get(row, col);
                transposed.set(col, row, *value); // Swap row and column positions
            }
        }
        transposed
    }
}

impl DenseMatrix<f64> {
    /// Performs dense * dense matrix multiplication.
    pub fn matmul(&self, other: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        assert_eq!(
            self.columns, other.rows,
            "The number of columns of the first matrix must match the number of rows of the second matrix"
        );

        // Create a new matrix to store the result.
        let mut result = DenseMatrix::new(self.rows, other.columns);

        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum = 0.0;
                for k in 0..self.columns {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }
}

impl DenseMatrix<f64> {
    /// Performs forward substitution to solve Lx = b.
    /// The matrix L must be lower triangular.
    pub fn forward_substitution(&self, b: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        assert_eq!(self.rows, self.columns, "The matrix must be square");
        assert_eq!(self.rows, b.data.len(), "The length of b must match the rows of the matrix");

        let mut x = vec![0.0; self.rows];

        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.get(i, j) * x[j];
            }
            let value = b.data[i] - sum;
            let diag = self.get(i, i);
            assert_ne!(*diag, 0.0, "The diagonal element cannot be zero");
            x[i] = value / diag;
        }

        DenseMatrix {
            rows: x.len(),
            columns: 1,
            data: x,
        }
    }
}

impl DenseMatrix<f64> {
    /// Performs backward substitution to solve Ux = b.
    /// The matrix U must be upper triangular.
    pub fn backward_substitution(&self, b: &Vec<f64>) -> Vec<f64> {
        assert_eq!(self.rows, self.columns, "The matrix must be square");
        assert_eq!(self.rows, b.len(), "The length of b must match the rows of the matrix");

        let mut x = vec![0.0; self.rows];

        for i in (0..self.rows).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..self.columns {
                sum += self.get(i, j) * x[j];
            }
            let value = b[i] - sum;
            let diag = self.get(i, i);
            assert_ne!(*diag, 0.0, "The diagonal element cannot be zero");
            x[i] = value / diag;
        }

        x
    }
}

impl DenseMatrix<f64> {
    // Calculates the maximum norm between two vectors.
    pub fn max_norm(&self, other: &DenseMatrix<f64>) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    // Calculates the Euclidean norm between two vectors.
    pub fn euclidean_norm(&self, other: &DenseMatrix<f64>) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// Carlos Alberto Escobedo Lopez

use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};
use std::fmt::Debug;
use num_traits::{Float, Zero};

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
    
    let matrixf32: Matrix<f32> = convert_file_csr::<f32>(lines.clone());

    let matrixf64: Matrix<f64> = convert_file_csr::<f64>(lines);

    let matrixf64 = matrixf64.to_dense();

    let matrixf32 = matrixf32.to_dense();

    let matrix64_conj = matrixf64.transpose_conjugate();

    let matrix32_conj = matrixf32.transpose_conjugate();

    let Bf32 = matrixf32.matmul(&matrix32_conj);

    let Bf64 = matrixf64.matmul(&matrix64_conj);

    let L32 = Bf32.cholesky().expect("Problem with decomposition");

    let L_T32 = L32.transpose_conjugate();


    let L64 = Bf64.cholesky().expect("Problem with decomposition");

    let L_T64 = L64.transpose_conjugate();

    

    let b32 = DenseMatrix {
        rows: L32.rows,
        columns: 1,
        data: vec![size.parse::<f32>().unwrap(); L32.rows],
    };

    let b64 = DenseMatrix {
        rows: L64.rows,
        columns: 1,
        data: vec![size.parse::<f64>().unwrap(); L64.rows],
    };

    let y32 = L32.forward_substitution(&b32);

    let y64 = L64.forward_substitution(&b64);
    
    let x32 = L_T32.backward_substitution(&y32);

    let x64 = L_T64.backward_substitution(&y64);

    let error_32 = matrixf32.matmul(&x32).subtract(&b32);

    let error_64 = matrixf64.matmul(&x64).subtract(&b64);

    let m_norm = error_64.max_norm(&b64);

    let eu_norm = error_64.euclidean_norm(&b64);

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

    // Primero obtenemos las dimensiones de la matriz
    let sizes: Vec<usize> = lines[0]
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let rows = sizes[0];
    let columns = sizes[1];
    let nnz = sizes[2];

    row_ptr = vec![0; rows + 1];  // row_ptr tiene un tamaño de rows + 1, como en CSR.

    // Mantener un contador de elementos por fila
    let mut row_counts = vec![0; rows];

    // Procesar las filas de datos
    for line in &lines[1..] {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        
        // Asegurarnos de que los índices sean al menos 1 para evitar desbordamiento
        let row = tokens[0].parse::<usize>().unwrap();
        let col = tokens[1].parse::<usize>().unwrap();

        if row < 1 || col < 1 {
            panic!("Índice inválido: fila = {}, columna = {}", row, col);
        }

        // Ajuste de índice a 0
        let row = row - 1;  // Ajuste de índice de fila
        let col = col - 1;  // Ajuste de índice de columna

        // Verificación de desbordamiento
        if row >= rows || col >= columns {
            panic!("Índice fuera de rango: fila = {}, columna = {}", row, col);
        }

        let value = tokens[2].parse::<T>().unwrap();

        values.push(value);
        col_indices.push(col);
        row_counts[row] += 1;
    }

    // Ahora construimos row_ptr acumulando el número de elementos por fila.
    for i in 1..=rows {
        row_ptr[i] = row_ptr[i - 1] + row_counts[i - 1];
    }

    // Verificación final para row_ptr
    if row_ptr[rows] != nnz {
        panic!("El número total de elementos no coincide con nnz. row_ptr[{}] = {}", rows, row_ptr[rows]);
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




impl<T> Matrix<T>
where
    T: Clone + Default + Copy,
{
    pub fn to_dense(&self) -> DenseMatrix<T> {
        let mut dense = DenseMatrix::new(self.rows, self.columns);

        for row in 0..self.rows {
            for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_indices[idx];  // Asumiendo que ya están en base 0
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




// Pruebas con `cargo test`.
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

        // Verificar las dimensiones de la matriz
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.columns, 3);

        // Verificar que todos los valores iniciales son 0.0 (valor por defecto para f32)
        for i in 0..matrix.rows {
            for j in 0..matrix.columns {
                assert_eq!(*matrix.get(i, j), 0.0);
            }
        }
    }



    #[test]
    fn test_set() {
        let mut matrix: DenseMatrix<f32> = DenseMatrix::new(3, 3);

        // Establecer algunos valores
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 3.0);

        // Verificar que los valores fueron correctamente asignados
        assert_eq!(*matrix.get(0, 0), 1.0);
        assert_eq!(*matrix.get(1, 1), 2.0);
        assert_eq!(*matrix.get(2, 2), 3.0);

        // Verificar que las demás posiciones siguen siendo 0.0
        assert_eq!(*matrix.get(0, 1), 0.0);
        assert_eq!(*matrix.get(1, 0), 0.0);
        assert_eq!(*matrix.get(2, 1), 0.0);
    }


    #[test]
    fn test_get() {
        let mut matrix: DenseMatrix<f32> = DenseMatrix::new(3, 3);

        // Establecer algunos valores
        matrix.set(0, 0, 5.0);
        matrix.set(1, 1, 10.0);
        matrix.set(2, 2, 15.0);

        // Verificar que el valor de las posiciones específicas es correcto
        assert_eq!(*matrix.get(0, 0), 5.0);
        assert_eq!(*matrix.get(1, 1), 10.0);
        assert_eq!(*matrix.get(2, 2), 15.0);

        // Verificar que obtener una posición no establecida da el valor por defecto (0.0)
        assert_eq!(*matrix.get(0, 1), 0.0);
    }


    #[test]
    fn test_to_dense() {
        // Crear una matriz dispersa manualmente (estructura Sparse Matrix)
        let sparse_matrix = Matrix {
            rows: 3,
            columns: 3,
            nnz: 3,  // Número de elementos no nulos
            values: vec![1.0, 2.0, 3.0],  // Los valores no nulos
            col_indices: vec![0, 1, 2],  // Los índices de columna
            row_ptr: vec![0, 1, 2, 3],  // Apuntadores de fila (último elemento es la cantidad de nnz)
        };

        // Convertir la matriz dispersa a una matriz densa
        let dense_matrix = sparse_matrix.to_dense();

        // Verificar las dimensiones
        assert_eq!(dense_matrix.rows, 3);
        assert_eq!(dense_matrix.columns, 3);


        // Verificar que los valores fueron correctamente copiados en la matriz densa
        assert_eq!(*dense_matrix.get(0, 0), 1.0);  // Primer valor no nulo
        assert_eq!(*dense_matrix.get(1, 1), 2.0);  // Segundo valor no nulo
        assert_eq!(*dense_matrix.get(2, 2), 3.0);  // Tercer valor no nulo





        // Verificar que las otras posiciones no modificadas sigan siendo None (o 0.0)
        assert_eq!(*dense_matrix.get(0, 1), 0.0);  // Cero implícito
        assert_eq!(*dense_matrix.get(0, 2), 0.0);  // Cero implícito
        assert_eq!(*dense_matrix.get(1, 0), 0.0);  // Cero implícito
        assert_eq!(*dense_matrix.get(2, 0), 0.0);  // Cero implícito
        assert_eq!(*dense_matrix.get(2, 1), 0.0);  // Cero implícito

    }




}
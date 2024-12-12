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
    
    let matrixf32: Matrix<f32> = convert_file_csr::<f32>(lines.clone());

    let matrix64: Matrix<f64> = convert_file_csr::<f64>(lines);

    let matrix64 = matrix64.to_dense();
    

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



pub fn convert_file_csr<T>(lines: Vec<String>) -> Matrix<T>
where
    T: FloatConvert + Default + Clone,
{
    let mut values = Vec::new();
    let mut col_indices = Vec::new();

    let line1 = &lines[0];
    let sizes: Vec<usize> = line1.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let rows = sizes[0];
    let columns = sizes[1];
    let nnz = sizes[2];

    let mut row_ptr = vec![0; rows + 1];
    let mut row_count = vec![0; rows];

    for line in lines.iter() {
        let entries: Vec<&str> = line.split_whitespace().collect();
        values.push(T::from_f64(entries[2].parse::<f64>().expect("Error casting")));
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




impl<T> Matrix<T>
where
    T: Clone + Default + Copy,
{
    pub fn to_dense(&self) -> DenseMatrix<T> {
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


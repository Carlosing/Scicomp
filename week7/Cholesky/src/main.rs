use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};
use std::fmt::Debug;

fn main() {
    let (file_name, size) = get_args().expect("Error getting args");

    let file_path = get_path(&file_name);

    let file = open_file(&file_path).expect("Error opening file");

    let reader = io::BufReader::new(file).lines();

    let lines: Vec<String> = reader.collect::<Result<_, _>>().expect("Error reading lines");

    

    let zero_vector = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    let one_vector = vec![0, 1, 1, 2, 0, 1, 2, 3, 1, 2, 3, 2];

    let two_vector = vec![0, 2, 4, 6, 8];

    let matriz1 = Matrix {
        rows: 4,
        columns: 4,
        nnz: 11,
        values: zero_vector,
        col_indices: one_vector,
        row_ptr: two_vector.clone(),
    };

    let matriz2 = convert_file_csr(lines);

    let sum = matriz2.add(&matriz2).expect("Failed to add matrices");

    let dis_mat = DenseMatrix {
        rows:2,
        columns:2,
        data: two_vector,
    };

    println!("{:?}", sum.values)
    


    
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





fn convert_file_csr(lines: Vec<String>) -> Matrix<i32> {
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


impl<T: std::ops::Add<Output = T> + Copy> Matrix<T> {
    pub fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>,String> {

        if self.rows != other.rows || self.columns != other.columns {
            return Err("Matrix dimensions must match".to_string());
        } else {
            let cart_prod = self.values.iter().zip(&other.values);

            let suma: Vec<T> = cart_prod.map(|(a,b)| *a+*b).collect();

            Ok(Matrix{
            rows : self.rows,
            columns : self.columns,
            nnz : suma.len(),
            values: suma,
            col_indices: self.col_indices.clone(),
            row_ptr: self.row_ptr.clone(),
            })

        }
        
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
    /// Crea una nueva matriz densa con valores por defecto.
    pub fn new(rows: usize, columns: usize) -> Self {
        let data = vec![T::default(); rows * columns];
        DenseMatrix { rows, columns, data }
    }

    /// Establece un valor en la posición (fila, columna).
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.columns, "Índices fuera de rango");
        self.data[row * self.columns + col] = value;
    }

    /// Obtiene un valor en la posición (fila, columna).
    pub fn get(&self, row: usize, col: usize) -> &T {
        assert!(row < self.rows && col < self.columns, "Índices fuera de rango");
        &self.data[row * self.columns + col]
    }
}

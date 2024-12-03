use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};

fn main() {
    let (file_name, size) = get_args().expect("Error getting args");

    let file_path = get_path(&file_name);

    let file = open_file(&file_path).expect("Error opening file");

    let reader = io::BufReader::new(file).lines();

    let lines: Vec<String> = reader.collect::<Result<_, _>>().expect("Error reading lines");

    let line1 = &lines[0];

    let sizes: Vec<usize> = line1.split_whitespace().map(|x| x.parse().unwrap()).collect(); 

    let (rows, columns, nnz) = (sizes[0], sizes[1], sizes[2]);

    let matriz1 = Matrix::new(rows, columns, nnz);

    let matriz2 = Matrix::new(2,3,4);

    



    
}


struct Matrix {
    rows : usize,
    columns : usize,
    nnz : usize,
}

impl Matrix{
    fn new(rows : usize, columns : usize, nnz: usize) -> Self {
        Matrix {rows, columns, nnz}
    }
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
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

    

    let zero_vector = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    let one_vector = vec![0, 1, 1, 2, 0, 1, 2, 3, 1, 2, 3, 2];

    let two_vector = vec![0, 2, 4, 6, 8];

    let matriz1 = Matrix {
        rows: 4,
        columns: 4,
        nnz: 11,
        values: zero_vector,
        col_indices: one_vector,
        row_ptr: two_vector,
    };

    let matriz2 = convert_file_CSR(lines);
    
    


    
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



// fn convert_file_CSR(lines: Vec<String>) -> (Vec<i32>, Vec<usize>, Vec<usize>) {
//     let mut values = Vec::new();
//     let mut col_indices = Vec::new();
//     let mut row_ptr = Vec::new();
//     let mut current_row = 0;

//     row_ptr.push(0);

//     for line in lines.iter().skip(1) {
//         let entries: Vec<&str> = line.split_whitespace().collect();
//         for entry in entries {
//             let parts: Vec<&str> = entry.split(':').collect();
//             let col_index: usize = parts[0].parse().unwrap();
//             let value: i32 = parts[1].parse().unwrap();
//             values.push(value);
//             col_indices.push(col_index);
//         }
//         current_row += 1;
//         row_ptr.push(values.len());
//     }

//     (values, col_indices, row_ptr)
// }



fn convert_file_CSR(lines: Vec<String>) -> (Vec<i32>, Vec<usize>, Vec<usize>, usize, usize, usize) {

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    
    let mut current_row = 0;
    

    

    let line1 = &lines[0];

    let sizes: Vec<usize> = line1.split_whitespace().map(|x| x.parse().unwrap()).collect(); 

    let rows = sizes[0];

    let mut row_ptr = vec![0; rows+1];

    let mut row_count = vec![0; rows];

    for line in lines.iter().skip(1) {

        let entries: Vec<&str> = line.split_whitespace().collect();

        values.push((entries[2].parse().expect("Error casting")));

        let mut row: usize = entries[0].parse().expect("Error casting");

        col_indices.push(entries[1].parse().expect("Error casting"));

        row_count[row-1] += 1;

        

    }

    for i in 1..=rows {
        row_ptr[i] = row_count[i-1] + row_ptr[i-1];
    }

    println!("{:?}", row_ptr);

    println!("{:?}", values);

    (values, col_indices, row_ptr, sizes[0], sizes[1], sizes[2])

}




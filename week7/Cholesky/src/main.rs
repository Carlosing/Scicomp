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

    


    let mut matrix = convert_file_csr(lines);

    let matrix = matrix.to_dense();

    let matrix_conj = matrix.transpose_conjugate();

    let B = matrix.matmul(&matrix_conj);

    let L = B.cholesky().expect("Problem with decomp");

    let L_T = L.transpose_conjugate();

    let decomp_prod = L.matmul(&L_T);

    let b = DenseMatrix{
        rows:decomp_prod.rows,
        columns: 1,
        data: vec![size.parse::<f64>().unwrap(); decomp_prod.rows],};

    let x = decomp_prod.forward_substitution(&b);

    let error = B.matmul(&x);

    let m_norm = error.max_norm(&b);

    println!("{}", m_norm);

    let eu_norm = error.euclidean_norm(&b);

    println!("{}", eu_norm);

    println!("{:?}", b.data);

    


    
    
    
    


    
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





fn convert_file_csr(lines: Vec<String>) -> Matrix<f64> {
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
    /// Realiza la descomposición de Cholesky para la matriz densa.
    /// Devuelve una nueva matriz triangular inferior `L` tal que A = L * Lᵀ.
    pub fn cholesky(&self) -> Result<DenseMatrix<f64>, String> {
        if self.rows != self.columns {
            return Err("La matriz debe ser cuadrada para realizar la descomposición de Cholesky".to_string());
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
                            "La matriz no es definida positiva. Encontrado un valor negativo o cero en la diagonal en la posición ({}, {})",
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
    /// Devuelve la traspuesta de la matriz (conjugada para números complejos).
    pub fn transpose_conjugate(&self) -> DenseMatrix<f64> {
        let mut transposed = DenseMatrix::new(self.columns, self.rows); // Cambia filas y columnas
        for row in 0..self.rows {
            for col in 0..self.columns {
                let value = self.get(row, col);
                transposed.set(col, row, *value); // Intercambia las posiciones fila y columna
            }
        }
        transposed
    }
}





impl DenseMatrix<f64> {
    /// Realiza la multiplicación de matrices densa * densa.
    pub fn matmul(&self, other: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        assert_eq!(
            self.columns, other.rows,
            "El número de columnas de la primera matriz debe coincidir con el número de filas de la segunda matriz"
        );

        // Crear una nueva matriz para almacenar el resultado.
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
    /// Realiza sustitución hacia adelante para resolver Lx = b.
    /// La matriz L debe ser triangular inferior.
    pub fn forward_substitution(&self, b: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        assert_eq!(self.rows, self.columns, "La matriz debe ser cuadrada");
        assert_eq!(self.rows, b.data.len(), "La longitud de b debe coincidir con las filas de la matriz");

        let mut x = vec![0.0; self.rows];

        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.get(i, j) * x[j];
            }
            let value = b.data[i] - sum;
            let diag = self.get(i, i);
            assert_ne!(*diag, 0.0, "El elemento diagonal no puede ser cero");
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
    /// Realiza sustitución hacia atrás para resolver Ux = b.
    /// La matriz U debe ser triangular superior.
    pub fn backward_substitution(&self, b: &Vec<f64>) -> Vec<f64> {
        assert_eq!(self.rows, self.columns, "La matriz debe ser cuadrada");
        assert_eq!(self.rows, b.len(), "La longitud de b debe coincidir con las filas de la matriz");

        let mut x = vec![0.0; self.rows];

        for i in (0..self.rows).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..self.columns {
                sum += self.get(i, j) * x[j];
            }
            let value = b[i] - sum;
            let diag = self.get(i, i);
            assert_ne!(*diag, 0.0, "El elemento diagonal no puede ser cero");
            x[i] = value / diag;
        }

        x
    }
}




impl DenseMatrix<f64> {
    // Calcula la norma máxima entre dos vectores.
    pub fn max_norm(&self, other: &DenseMatrix<f64>) -> f64 {
        assert_eq!(self.rows, other.rows, "El tamaño del vector debe coincidir con las filas de la matriz");

        // Calculamos el valor absoluto del valor más grande entre los elementos de los dos vectores
        other.data.iter().map(|&x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0)
    }

    // Calcula la norma euclidiana entre dos vectores.
    pub fn euclidean_norm(&self, other: &DenseMatrix<f64>) -> f64 {
        assert_eq!(self.rows, other.rows, "El tamaño del vector debe coincidir con las filas de la matriz");

        // Sumamos los cuadrados de los elementos y sacamos la raíz cuadrada del resultado
        let sum_of_squares: f64 = other.data.iter().map(|&x| x * x).sum();
        sum_of_squares.sqrt()
    }
}




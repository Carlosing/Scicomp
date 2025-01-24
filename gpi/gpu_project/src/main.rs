fn main() {
    let matriz1  =  Matrix::new(1,2,vec![1.1, 2.2]);

    print!("{}", matriz1.get(0,0));
}


struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}




impl Matrix {
    fn new(rows: usize, cols:usize, data:Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Incorrect dimensions");
        Matrix {
            rows, 
            cols, 
            data
        }
    }


    fn zeros(rows: usize, cols:usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows*cols],
        }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row*self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_new() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Incorrect dimensions")]
    fn test_matrix_new_incorrect_dimensions() {
        Matrix::new(2, 2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matrix_zeros() {
        let matrix = Matrix::zeros(2, 2);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![0.0; 4]);
    }

    #[test]
    fn test_matrix_get() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_set() {
        let mut matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        matrix.set(0, 0, 5.0);
        assert_eq!(matrix.get(0, 0), 5.0);
    }
}

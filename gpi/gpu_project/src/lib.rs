use rayon::prelude::*;



pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}




impl Matrix {
    pub fn new(rows: usize, cols:usize, data:Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Incorrect dimensions");
        Matrix {
            rows, 
            cols, 
            data
        }
    }


    pub fn zeros(rows: usize, cols:usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows*cols],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row*self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Incompatible matrices for multiplication");

        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }
}


impl Matrix {
    pub fn multiply_parallel(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Incompatible matrices for multiplication");

        let mut result = Matrix::zeros(self.rows, other.cols);

        result.data.par_iter_mut().enumerate().for_each(|(index, value)| {
            let row = index / other.cols;
            let col = index % other.cols;
            let mut sum = 0.0;
            for k in 0..self.cols {
                sum += self.get(row, k) * other.get(k, col);
            }
            *value = sum;
        });

        result
    }
}

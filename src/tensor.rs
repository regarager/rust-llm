use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let strides = Self::compute_strides(shape);

        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(data.len(), size);
        let strides = Self::compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    pub fn resize(&mut self, shape: &[usize]) {
        let size: usize = shape.iter().product();
        assert_eq!(self.data.len(), size);

        self.shape = shape.to_vec();
        self.strides = Self::compute_strides(shape);
    }

    pub fn offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());

        let mut offset = 0;

        for (i, &dim) in indices.iter().enumerate() {
            assert!(dim < self.shape[i]);
            offset += dim * self.strides[i];
        }

        offset
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn to_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn to_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl Index<&[usize]> for Tensor {
    type Output = f32;
    fn index(&self, indices: &[usize]) -> &f32 {
        &self.data[self.offset(indices)]
    }
}

impl IndexMut<&[usize]> for Tensor {
    fn index_mut(&mut self, indices: &[usize]) -> &mut f32 {
        let off = self.offset(indices);
        &mut self.data[off]
    }
}

pub fn outer(x: &[f32], y: &[f32]) -> Tensor {
    let n = x.len();
    let m = y.len();

    let mut mat = Tensor::new(&[n, m]); 
    let slice = mat.to_mut_slice();

    for i in 0..n {
        for j in 0..m {
            slice[i * m + j] = x[i] * y[j];
        }
    }

    mat
}

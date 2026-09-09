use crate::tensor::Tensor;

pub struct Embedding {
    pub vocab_size: usize,
    pub d_model: usize,
    pub embed: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Embedding {
            vocab_size,
            d_model,
            embed: Tensor::new(&[vocab_size, d_model]),
        }
    }

    pub fn forward(&self, x: Vec<i32>) -> Tensor {
        // TODO: change x to a tensor and/or allow batching
        let mut embeddings = Tensor::new(&[x.len(), self.d_model]);
        let target = embeddings.to_mut_slice();
        let emb = self.embed.to_slice();

        for (i, token) in x.iter().enumerate() {
            let t = *token as usize;
            target[(i * self.d_model)..((i + 1) * self.d_model)]
                .copy_from_slice(&emb[(t * self.d_model)..((t + 1) * self.d_model)]);
        }

        embeddings
    }
}

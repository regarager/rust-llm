use std::fs;

use crate::tokenizer::{Tokenizer, TokenizerConfig};

mod tokenizer;

fn main() {
    let mut tokenizer = Tokenizer::new(TokenizerConfig { vocab_size: 1257 });
    let text = fs::read_to_string("data/short.txt").unwrap();
    let bytes = text.as_bytes().to_vec();
    tokenizer.load_enc(&bytes);
    println!();
    let sample = bytes.clone();
    let enc = tokenizer.encode(&sample);
    let dec = tokenizer.decode(&enc);
    println!("sample length: {}", sample.len());
    println!("encoded length: {}", enc.len());
    println!("{}", sample == dec);
}

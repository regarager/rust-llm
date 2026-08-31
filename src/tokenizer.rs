use std::collections::HashMap;

#[derive(Debug, Default)]
struct TrieNode {
    pub children: HashMap<u8, TrieNode>,
    pub value: Option<i32>,
}

struct Trie {
    root: TrieNode,
    size: usize,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            root: TrieNode::default(),
            size: 0,
        }
    }

    #[allow(dead_code)]
    pub fn len(self) -> usize {
        self.size
    }

    #[allow(dead_code)]
    pub fn entry(&self, seq: &[u8]) -> Option<&TrieNode> {
        let mut curr = &self.root;

        for i in seq {
            curr = curr.children.get(i)?;
        }

        Some(curr)
    }

    #[allow(dead_code)]
    pub fn get(&self, seq: &[u8]) -> Option<i32> {
        self.entry(seq)?.value
    }

    pub fn insert(&mut self, seq: &[u8]) {
        let mut node = &mut self.root;

        for &i in seq {
            node = node.children.entry(i).or_default();
        }

        node.value = Some(self.size as i32);

        self.size += 1;
    }

    pub fn root(&self) -> &TrieNode {
        &self.root
    }
}

pub struct Encoder {
    vocab: Trie,
    reverse: HashMap<i32, Vec<u8>>,
}

impl Encoder {
    pub fn new() -> Self {
        Encoder {
            vocab: Trie::new(),
            reverse: HashMap::new(),
        }
    }
}

pub struct TokenizerConfig {
    pub vocab_size: usize,
}

pub struct Tokenizer {
    config: TokenizerConfig,
    enc: Encoder,

    #[allow(dead_code)]
    eos_token: String,
}

impl Tokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Tokenizer {
            config,
            enc: Encoder::new(),
            eos_token: String::from("<|endoftext|>"),
        }
    }

    pub fn load_enc(&mut self, text: &[u8]) {
        for i in 0..=255 {
            self.enc.vocab.insert(&[i]);
            self.enc.reverse.insert(i as i32, vec![i]);
        }

        for step in 0..(self.config.vocab_size - 257) {
            let seq = self.encode(text);

            let mut cnt: HashMap<(i32, i32), i32> = HashMap::new();

            for i in 0..seq.len() - 1 {
                cnt.entry((seq[i], seq[i + 1]))
                    .and_modify(|x| *x += 1)
                    .or_insert(1);
            }

            let mut best_pair = (0, 0);
            let mut best = 0;
            for it in cnt.iter() {
                if *it.1 > best {
                    best_pair = *it.0;
                    best = *it.1;
                }
            }

            if best == 1 {
                println!("warning: stopping byte-pair encoding early due to maximal compression");
                break;
            }

            let mut new_token = self.enc.reverse[&best_pair.0].clone();
            new_token.extend_from_slice(&self.enc.reverse[&best_pair.1]);

            self.enc.vocab.insert(&new_token);
            self.enc
                .reverse
                .insert((step + 256).try_into().unwrap(), new_token.clone());

            dbg!(step);
        }
    }

    pub fn encode(&self, text: &[u8]) -> Vec<i32> {
        let mut res = Vec::new();
        let mut pos = 0;
        let root = self.enc.vocab.root();

        while pos < text.len() {
            let mut curr = root;
            let mut last_match = None;
            let mut next_pos = pos;

            while next_pos < text.len() {
                if let Some(child) = curr.children.get(&text[next_pos]) {
                    curr = child;
                    next_pos += 1;
                    if let Some(value) = child.value {
                        last_match = Some((value, next_pos));
                    }
                } else {
                    break;
                }
            }

            if let Some((value, end_pos)) = last_match {
                res.push(value);
                pos = end_pos;
            } else {
                if let Some(byte_node) = root.children.get(&text[pos]) {
                    if let Some(value) = byte_node.value {
                        res.push(value);
                        pos += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        res
    }

    pub fn decode(&self, seq: &[i32]) -> Vec<u8> {
        let mut res = Vec::new();

        for i in seq {
            res.extend_from_slice(&self.enc.reverse[i]);
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn enc_dec() {
        let mut tokenizer = Tokenizer::new(TokenizerConfig { vocab_size: 1257 });
        let text = fs::read_to_string("data/short.txt").unwrap();
        let bytes = text.as_bytes().to_vec();
        tokenizer.load_enc(&bytes);
        let sample = bytes.clone();
        let enc = tokenizer.encode(&sample);
        let dec = tokenizer.decode(&enc);
        assert_eq!(sample, dec);
        assert_eq!(tokenizer.enc.vocab.len(), tokenizer.config.vocab_size);
    }
}

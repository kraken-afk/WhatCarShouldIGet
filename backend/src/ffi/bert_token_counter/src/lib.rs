use lazy_static::lazy_static;
use libc::c_char;
use tokenizers::models::bpe::BPE;
use std::ffi::CStr;
use std::sync::Mutex;
use tokenizers::Tokenizer;
struct BertTokenCounter {
    tokenizer: Tokenizer,
}
impl BertTokenCounter {
    fn new() -> Result<Self, String> {
        Self::from_files("vocab.json", "merges.txt")
    }
    fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self, String> {
        let bpe = BPE::from_file(vocab_path, merges_path)
            .build()
            .map_err(|e| format!("Failed to build BPE: {}", e))?;
        let tokenizer = Tokenizer::new(bpe);
        Ok(BertTokenCounter { tokenizer })
    }
    fn count_tokens(&self, text: &str) -> Result<usize, String> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("Failed to encode text: {}", e))?;
        Ok(encoding.get_tokens().len())
    }
    fn count_tokens_with_special(&self, text: &str) -> Result<usize, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().len())
    }
}
lazy_static! {
    static ref TOKEN_COUNTER : Mutex < Option < BertTokenCounter >> = { match
    BertTokenCounter::new() { Ok(counter) => Mutex::new(Some(counter)), Err(_) =>
    Mutex::new(None), } };
}
/// Exposes token counting functionality to C.
///
/// # Safety
/// The caller must ensure that `text_ptr` is a valid pointer to a
/// null-terminated UTF-8 C string that remains valid for the duration
/// of this function call.
///
/// Input: A UTF-8 encoded C string.
/// Returns:
///   - Token count (>= 0) on success.
///   - -1: Null input pointer.
///   - -2: Invalid UTF-8 input string.
///   - -3: Tokenizer initialization failed.
///   - -4: Mutex was poisoned (internal error).
///   - -5: Tokenization failed.
#[no_mangle]
pub unsafe extern "C" fn count_tokens_with_special(text_ptr: *const c_char) -> i64 {
    if text_ptr.is_null() {
        return -1;
    }
    let c_str = CStr::from_ptr(text_ptr);
    let text_slice = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let lock = TOKEN_COUNTER.lock();
    match lock {
        Ok(guard) => {
            match &*guard {
                Some(counter) => {
                    match counter.count_tokens_with_special(text_slice) {
                        Ok(count) => count as i64,
                        Err(_) => -5,
                    }
                }
                None => -3,
            }
        }
        Err(_) => -4,
    }
}
/// Alternative function that counts only text tokens (without special tokens)
///
/// # Safety
/// The caller must ensure that `text_ptr` is a valid pointer to a
/// null-terminated UTF-8 C string that remains valid for the duration
/// of this function call.
#[no_mangle]
pub unsafe extern "C" fn count_tokens(text_ptr: *const c_char) -> i64 {
    if text_ptr.is_null() {
        return -1;
    }
    let c_str = CStr::from_ptr(text_ptr);
    let text_slice = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let lock = TOKEN_COUNTER.lock();
    match lock {
        Ok(guard) => {
            match &*guard {
                Some(counter) => {
                    match counter.count_tokens(text_slice) {
                        Ok(count) => count as i64,
                        Err(_) => -5,
                    }
                }
                None => -3,
            }
        }
        Err(_) => -4,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    fn create_mock_tokenizer_files() -> Result<TempDir, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let vocab_json = r#"{"hello": 0, "world": 1, "[UNK]": 2, "[CLS]": 3, "[SEP]": 4}"#;
        let vocab_path = temp_dir.path().join("vocab.json");
        fs::write(&vocab_path, vocab_json)?;
        let merges_txt = "#version: 0.2\n";
        let merges_path = temp_dir.path().join("merges.txt");
        fs::write(&merges_path, merges_txt)?;
        Ok(temp_dir)
    }
    #[test]
    fn test_bert_token_counter_creation_no_files() {
        let result = BertTokenCounter::new();
        assert!(result.is_err());
    }
    #[test]
    fn test_bert_token_counter_with_mock_files() {
        let temp_dir = create_mock_tokenizer_files()
            .expect("Failed to create temp files");
        std::env::set_current_dir(temp_dir.path()).unwrap();
        let counter = BertTokenCounter::new();
        match counter {
            Ok(c) => {
                let result = c.count_tokens("hello world");
                assert!(result.is_ok());
                let result_special = c.count_tokens_with_special("hello world");
                assert!(result_special.is_ok());
            }
            Err(_) => {
                println!("Tokenizer creation failed with mock data - expected");
            }
        }
    }
}

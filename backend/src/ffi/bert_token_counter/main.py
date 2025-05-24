import ctypes
import os
import platform

# Determine the project root relative to this script or assume a structure.
# For this example, we'll construct the path assuming the script is run
# from the project's root directory.
# If your script is elsewhere, you might need to adjust ROOT_DIR.
# e.g., ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT_DIR = '.'  # Assuming script is run from project root

LIB_BASE_PATH = os.path.join(
    ROOT_DIR, 'src', 'ffi', 'bert_token_counter', 'target', 'release'
)

# Load the Rust library
system = platform.system()
if system == 'Linux':
    lib_name = 'libbert_token_counter.so'
elif system == 'Darwin':  # macOS
    lib_name = 'libbert_token_counter.dylib'
elif system == 'Windows':
    lib_name = 'bert_token_counter.dll'
else:
    print(f'Unsupported platform: {system}')
    exit(1)

lib_path = os.path.join(LIB_BASE_PATH, lib_name)

if not os.path.exists(lib_path):
    print(f'Library not found at {lib_path}')
    print('Build the library first with: cargo build --release')
    print(
        'Ensure it is in: <your_project_root>/src/bert_token_counter/target/release/'
    )
    exit(1)

# Load the shared library
lib = ctypes.CDLL(lib_path)

# Define the function signatures
# Corrected to match the function names called in your Python wrappers
lib.count_tokens.argtypes = [ctypes.c_char_p]
lib.count_tokens.restype = ctypes.c_int64  # Returns a 64-bit integer

lib.count_tokens_with_special.argtypes = [ctypes.c_char_p]
lib.count_tokens_with_special.restype = (
    ctypes.c_int64
)  # Returns a 64-bit integer


def count_tokens(text: str) -> int:
    """
    Count BERT tokens including special tokens [CLS], [SEP]

    Returns:
    - Token count (>= 0) on success
    - -1: Null input pointer
    - -2: Invalid UTF-8 input string
    - -3: Tokenizer initialization failed
    - -4: Mutex was poisoned (internal error)
    - -5: Tokenization failed
    """
    text_bytes = text.encode('utf-8')
    result: int = lib.count_tokens(text_bytes)
    return result


def count_tokens_with_special(text: str) -> int:
    """
    Count BERT tokens without special tokens

    Same error codes as count_bert_tokens
    """
    text_bytes = text.encode('utf-8')
    result: int = lib.count_bert_tokens_only(text_bytes)
    return result

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer

from src.ffi.bert_token_counter import main as token_counter
from src.libs.tokenizer import count_tokens


class TransformResult(TypedDict):
    """Generic result type independent of any specific vector database."""

    embeddings: list[list[float]]
    documents: list[str]
    metadatas: list[dict[str, str | int | float | bool | None]]
    ids: list[str]


class ChromaDBData(TypedDict):
    """ChromaDB-specific data format."""

    embeddings: npt.NDArray[np.int32 | np.float32]
    documents: list[str]
    metadatas: list[Mapping[str, str | int | float | bool | None]]
    ids: list[str]


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def create_transformer():
    from sentence_transformers import SentenceTransformer, models

    word_embedding_model = models.Transformer('microsoft/graphcodebert-base')

    word_embedding_model.tokenizer

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def parse_embeddings(embeddings: torch.Tensor) -> list[list[float]]:
    """Parses embeddings from a torch.Tensor to a list of lists of floats."""
    return embeddings.tolist()


def find_optimal_split(
    line_content: str, max_chunk_tokens: int, chunk_header: str
) -> int:
    """
    Find optimal split point using binary search to maximize content
    while staying within token limit.
    """
    left, right = 0, len(line_content)
    best_split = 0

    while left <= right:
        mid = (left + right) // 2
        test_content = line_content[:mid] + '\n'
        test_text = chunk_header + test_content

        token_count = count_tokens(test_text)

        if token_count <= max_chunk_tokens:
            best_split = mid
            left = mid + 1
        else:
            right = mid - 1

    # Ensure we split at word boundary when possible
    if best_split > 0 and best_split < len(line_content):
        # Look backwards for a space to split on word boundary
        for i in range(best_split, max(0, best_split - 50), -1):
            if line_content[i].isspace():
                return i + 1

    return max(1, best_split)  # Ensure at least 1 character is taken


def transform_file(
    file_path: str,
    transformer: SentenceTransformer,
    max_chunk_tokens: int = 512,
) -> TransformResult:
    """
    Transforms file content using dir-assistant-style chunking approach.
    Embeds only raw code content while storing headers and metadata separately.
    Returns data in a generic format independent of any vector database.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' does not exist.")

    raw_code_chunks: list[str] = []
    metadata_list = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines: list[str] = f.readlines()

    current_chunk: str = ''
    start_line_number: int = 1

    def _save_current_chunk(
        chunk_content: str, start_line: int, end_line: int
    ) -> None:
        """Helper function to save a chunk and its metadata."""
        chunk_header: str = (
            f"File '{file_path}' lines {start_line}-{end_line}:\n\n"
        )

        raw_code_chunks.append(chunk_content.rstrip('\n'))
        metadata_list.append(
            {
                'file_path': file_path,
                'start_line': str(start_line),
                'end_line': str(end_line),
                'chunk_index': str(len(raw_code_chunks) - 1),
                'chunk_header': chunk_header,
                'full_display_text': chunk_header + chunk_content,
            }
        )

    for line_number, line in enumerate(lines, start=1):
        line_content: str = line.rstrip('\n')

        while line_content:
            # Create chunk header for token counting (but don't include in final chunk)
            chunk_header: str = f"File '{file_path}' lines {start_line_number}-{line_number}:\n\n"
            proposed_chunk: str = current_chunk + line_content + '\n'
            proposed_text_with_header: str = chunk_header + proposed_chunk

            # Count tokens for the complete text (header + code) to respect limits
            chunk_tokens: int = count_tokens(proposed_text_with_header)

            if chunk_tokens <= max_chunk_tokens:
                current_chunk = proposed_chunk
                break
            else:
                if current_chunk == '':
                    # Split long line using binary search approach
                    split_point: int = find_optimal_split(
                        line_content,
                        max_chunk_tokens,
                        chunk_header,
                    )
                    current_chunk = line_content[:split_point] + '\n'
                    line_content = line_content[split_point:]
                else:
                    # Save current chunk (raw code only) and metadata separately
                    _save_current_chunk(
                        current_chunk, start_line_number, line_number - 1
                    )

                    current_chunk = ''
                    start_line_number = line_number

    # Handle remaining content
    if current_chunk:
        _save_current_chunk(current_chunk, start_line_number, len(lines))

    # Generate embeddings for raw code content only (no headers)
    full_texts = [meta['full_display_text'] for meta in metadata_list]
    embeddings_tensor = normalize_embeddings(
        transformer.encode(full_texts, convert_to_tensor=True)
    )
    embeddings: list[list[float]] = parse_embeddings(embeddings_tensor)

    # Generate unique IDs for each chunk
    ids: list[str] = [
        f'{file_path}:{meta["start_line"]}-{meta["end_line"]}-{meta["chunk_index"]}'
        for meta in metadata_list
    ]

    return TransformResult(
        embeddings=embeddings,
        documents=raw_code_chunks,  # Raw code only, no headers
        metadatas=metadata_list,  # Headers and display text stored here
        ids=ids,
    )


def parse_to_chromadb(result: TransformResult) -> ChromaDBData:
    """
    Converts generic TransformResult to ChromaDB-compatible format.
    """
    return ChromaDBData(
        embeddings=np.array(result['embeddings'], dtype=np.float32),
        documents=result['documents'],
        metadatas=[
            {k: v for k, v in metadata.items()}  # Convert to Mapping
            for metadata in result['metadatas']
        ],
        ids=result['ids'],
    )


def dir_walker(path: str):
    """Walks through the directory and yields all files."""
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist.")

    ignored = {'.git', '__pycache__', 'node_modules', '.vscode', '.idea'}
    try:
        with open('.gitignore', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ignored.add(line)
    except FileNotFoundError:
        pass

    ignored_media = {
        '.jpg',
        '.jpeg',
        '.png',
        '.gif',
        '.bmp',
        '.mp4',
        '.avi',
        '.mov',
        '.mkv',
        '.webm',
        '.mp3',
        '.wav',
        '.ogg',
    }
    binary_excluded_ext = {'.exe', '.dll', '.so', '.bin'}

    for root, dirs, files in os.walk(path):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ignored]
        for file in files:
            if file.startswith('.') or file in {'package-lock.json'}:
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in ignored_media or ext in binary_excluded_ext:
                continue
            file_path = os.path.join(root, file)
            if is_binary_file(file_path):
                continue
            yield file_path


@lru_cache(maxsize=None)
def is_binary_file(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
    except Exception:
        return True
    return False

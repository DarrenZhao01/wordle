"""Batched pattern matrix generation, Turn-1 entropy caching, and metadata validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.dictionary import DEFAULT_DATA_DIR, WordleDictionary
from src.engine import ENCODING_VERSION, NUM_PATTERNS, POWERS_OF_3


PATTERN_MATRIX_FILENAME = "pattern_matrix.npy"
OPENING_ENTROPIES_FILENAME = "opening_entropies.npy"
CACHE_META_FILENAME = "cache_meta.json"

CACHE_SCHEMA_VERSION = "1.0"
ENTROPY_VERSION = "shannon_base2_lut_float64_v1"


def words_to_char_codes(words: Sequence[str]) -> np.ndarray:
    """Convert a sequence of 5-letter lowercase ASCII words into a (N, 5) uint8 array in 0..25."""
    if not words:
        return np.empty((0, 5), dtype=np.uint8)
    joined = "".join(words).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(len(words), 5)
    return arr - ord("a")


def compute_pattern_matrix_batched(
    guess_words: Sequence[str],
    answer_words: Sequence[str],
    batch_size: int = 512,
) -> np.ndarray:
    """Compute the (N_guesses, N_answers) uint8 feedback matrix in guess batches."""
    g_codes = words_to_char_codes(guess_words)
    a_codes = words_to_char_codes(answer_words)

    num_g = len(guess_words)
    num_a = len(answer_words)
    matrix = np.empty((num_g, num_a), dtype=np.uint8)
    weights = np.array(POWERS_OF_3, dtype=np.uint8)

    for start in range(0, num_g, batch_size):
        end = min(start + batch_size, num_g)
        g_batch = g_codes[start:end]  # (B, 5)

        # Pass 1: exact positional matches (GREEN = 2)
        is_green = g_batch[:, None, :] == a_codes[None, :, :]  # (B, N_a, 5)
        trits = np.where(is_green, 2, 0).astype(np.uint8)

        # Track which positions in each answer have already been matched (Green or prior Yellow)
        answer_pos_used = is_green.copy()

        # Pass 2: left-to-right YELLOW (1) assignment for non-green guess positions
        for i in range(5):
            need_match = ~is_green[:, :, i]  # (B, N_a)
            g_col = g_batch[:, i][:, None]   # (B, 1)
            for j in range(5):
                can_match = (
                    need_match
                    & (~answer_pos_used[:, :, j])
                    & (g_col == a_codes[None, :, j])
                )
                trits[:, :, i] = np.where(can_match, 1, trits[:, :, i])
                answer_pos_used[:, :, j] |= can_match
                need_match &= ~can_match

        matrix[start:end] = np.sum(trits * weights[None, None, :], axis=2, dtype=np.uint8)

    return matrix


def compute_entropy_scalar_row(pattern_row: Sequence[int]) -> float:
    """Scalar reference calculation of base-2 Shannon entropy for a single guess's pattern row."""
    k = len(pattern_row)
    if k <= 1:
        return 0.0
    counts: Dict[int, int] = {}
    for p in pattern_row:
        ip = int(p)
        counts[ip] = counts.get(ip, 0) + 1
    entropy = 0.0
    for c in counts.values():
        prob = c / k
        entropy -= prob * math.log2(prob)
    return entropy


def compute_entropies(
    pattern_submatrix: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Compute base-2 Shannon entropy for each row in pattern_submatrix of shape (N_rows, K)."""
    num_rows, k = pattern_submatrix.shape
    if k <= 1:
        return np.zeros(num_rows, dtype=np.float64)

    # Precompute exact term lookup table for partition size n in 0..K
    n_vals = np.arange(k + 1, dtype=np.float64)
    term_lut = np.zeros(k + 1, dtype=np.float64)
    probs = n_vals[1:] / float(k)
    term_lut[1:] = -probs * np.log2(probs)

    entropies = np.empty(num_rows, dtype=np.float64)

    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = pattern_submatrix[start:end]
        b_len = end - start

        row_offsets = (np.arange(b_len, dtype=np.int32)[:, None] * NUM_PATTERNS) + batch
        counts = np.bincount(
            row_offsets.ravel(),
            minlength=b_len * NUM_PATTERNS,
        ).reshape(b_len, NUM_PATTERNS)

        entropies[start:end] = term_lut[counts].sum(axis=1)

    return entropies


def compute_entropies_from_transposed(
    pattern_submatrix_t: np.ndarray,
) -> np.ndarray:
    """Compute base-2 Shannon entropy for each column in pattern_submatrix_t of shape (K, N_rows)."""
    k, num_rows = pattern_submatrix_t.shape
    if k <= 1:
        return np.zeros(num_rows, dtype=np.float64)

    n_vals = np.arange(k + 1, dtype=np.float64)
    term_lut = np.zeros(k + 1, dtype=np.float64)
    probs = n_vals[1:] / float(k)
    term_lut[1:] = -probs * np.log2(probs)

    if k <= 80:
        s = np.sort(pattern_submatrix_t, axis=0)
        ent = np.zeros(num_rows, dtype=np.float64)
        run_len = np.ones(num_rows, dtype=np.int32)
        for r in range(1, k):
            same = s[r] == s[r - 1]
            ent += np.where(same, 0.0, term_lut[run_len])
            run_len = np.where(same, run_len + 1, 1)
        ent += term_lut[run_len]
        return ent

    return compute_entropies(np.ascontiguousarray(pattern_submatrix_t.T))


@dataclass(frozen=True)
class FeedbackCache:
    """Validated in-memory feedback pattern matrix and Turn-1 opening entropy vector."""

    pattern_matrix: np.ndarray     # shape (N_guesses, N_answers), uint8, C-contiguous
    pattern_matrix_t: np.ndarray   # shape (N_answers, N_guesses), uint8, C-contiguous
    opening_entropies: np.ndarray  # shape (N_guesses,), float64
    metadata: Dict[str, object]
    loaded_from_disk: bool
    load_or_build_time_ms: float


def _expected_cache_metadata(dictionary: WordleDictionary) -> Dict[str, object]:
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "encoding_version": ENCODING_VERSION,
        "entropy_version": ENTROPY_VERSION,
        "pattern_matrix_shape": [dictionary.num_guesses, dictionary.num_answers],
        "pattern_matrix_dtype": "uint8",
        "opening_entropies_shape": [dictionary.num_guesses],
        "opening_entropies_dtype": "float64",
    }
    meta.update(dictionary.metadata_dict())
    return meta


def is_cache_valid(
    dictionary: WordleDictionary,
    meta_path: Path,
    matrix_path: Path,
    entropies_path: Path,
) -> Tuple[bool, Optional[Dict[str, object]]]:
    """Check whether on-disk cache files exist and match all dictionary and version hashes."""
    if not (meta_path.exists() and matrix_path.exists() and entropies_path.exists()):
        return False, None

    try:
        stored_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, None

    expected = _expected_cache_metadata(dictionary)
    for key, expected_val in expected.items():
        if stored_meta.get(key) != expected_val:
            return False, None

    return True, stored_meta


def load_or_build_cache(
    dictionary: WordleDictionary,
    data_dir: Path = DEFAULT_DATA_DIR,
    force_rebuild: bool = False,
    batch_size: int = 512,
) -> FeedbackCache:
    """Load validated pattern_matrix and opening_entropies from disk, or build and save them."""
    t0 = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / CACHE_META_FILENAME
    matrix_path = data_dir / PATTERN_MATRIX_FILENAME
    entropies_path = data_dir / OPENING_ENTROPIES_FILENAME

    if not force_rebuild:
        valid, stored_meta = is_cache_valid(dictionary, meta_path, matrix_path, entropies_path)
        if valid and stored_meta is not None:
            matrix = np.load(matrix_path)
            entropies = np.load(entropies_path)
            if (
                matrix.shape == (dictionary.num_guesses, dictionary.num_answers)
                and matrix.dtype == np.uint8
                and entropies.shape == (dictionary.num_guesses,)
                and entropies.dtype == np.float64
            ):
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return FeedbackCache(
                    pattern_matrix=matrix,
                    pattern_matrix_t=np.ascontiguousarray(matrix.T),
                    opening_entropies=entropies,
                    metadata=stored_meta,
                    loaded_from_disk=True,
                    load_or_build_time_ms=elapsed_ms,
                )

    matrix = compute_pattern_matrix_batched(
        guess_words=dictionary.allowed_guesses,
        answer_words=dictionary.answers,
        batch_size=batch_size,
    )
    entropies = compute_entropies(matrix, batch_size=batch_size * 2)

    np.save(matrix_path, matrix)
    np.save(entropies_path, entropies)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    meta = _expected_cache_metadata(dictionary)
    meta["build_time_ms"] = round(elapsed_ms, 2)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return FeedbackCache(
        pattern_matrix=matrix,
        pattern_matrix_t=np.ascontiguousarray(matrix.T),
        opening_entropies=entropies,
        metadata=meta,
        loaded_from_disk=False,
        load_or_build_time_ms=elapsed_ms,
    )

"""Dictionary loading, SHA-256 hashing, and deterministic Dev/Test split management."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ANSWERS_FILENAME = "wordle-answers-alphabetical.txt"
ALLOWED_FILENAME = "wordle-allowed-guesses.txt"
SPLITS_FILENAME = "splits.json"
DEFAULT_SPLIT_SEED = 42
DEFAULT_DEV_RATIO = 0.20
SPLIT_SCHEMA_VERSION = "1.0"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_wordlist(words: List[str]) -> str:
    canonical = "\n".join(words).encode("utf-8")
    return _sha256_bytes(canonical)


def derive_game_seed(run_seed: int, target_word: str) -> int:
    """Derive a deterministic 64-bit unsigned integer seed from (run_seed, target_word)."""
    payload = f"{int(run_seed)}:{target_word.strip().lower()}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class WordleDictionary:
    """Holds canonical sorted answer and allowed-guess dictionaries and index mappings."""

    answers: Tuple[str, ...]
    allowed_guesses: Tuple[str, ...]
    answers_file_sha256: str
    allowed_file_sha256: str
    answers_list_sha256: str
    guesses_list_sha256: str
    answer_to_idx: Dict[str, int]
    guess_to_idx: Dict[str, int]
    answer_idx_to_guess_idx: np.ndarray  # shape (N_answers,), int32
    guess_idx_to_answer_idx: np.ndarray  # shape (N_guesses,), int32 (-1 if non-answer)
    is_answer_mask: np.ndarray           # shape (N_guesses,), bool
    dev_targets: Tuple[str, ...]
    test_targets: Tuple[str, ...]
    dev_set: Set[str]
    test_set: Set[str]
    split_seed: int

    @property
    def num_answers(self) -> int:
        return len(self.answers)

    @property
    def num_guesses(self) -> int:
        return len(self.allowed_guesses)

    def metadata_dict(self) -> Dict[str, object]:
        return {
            "num_answers": self.num_answers,
            "num_guesses": self.num_guesses,
            "answers_file_sha256": self.answers_file_sha256,
            "allowed_file_sha256": self.allowed_file_sha256,
            "answers_list_sha256": self.answers_list_sha256,
            "guesses_list_sha256": self.guesses_list_sha256,
            "first_answer": self.answers[0],
            "last_answer": self.answers[-1],
            "first_guess": self.allowed_guesses[0],
            "last_guess": self.allowed_guesses[-1],
            "split_seed": self.split_seed,
            "num_dev_targets": len(self.dev_targets),
            "num_test_targets": len(self.test_targets),
            "dev_targets_sha256": _sha256_wordlist(list(self.dev_targets)),
            "test_targets_sha256": _sha256_wordlist(list(self.test_targets)),
        }


def _validate_words(words: List[str], source_name: str) -> List[str]:
    cleaned: List[str] = []
    for raw in words:
        w = raw.strip().lower()
        if not w:
            continue
        if len(w) != 5 or not w.isalpha() or not w.isascii():
            raise ValueError(f"Invalid 5-letter ASCII word {w!r} in {source_name}")
        cleaned.append(w)
    return sorted(set(cleaned))


def _load_or_create_splits(
    answers: List[str],
    answers_list_sha256: str,
    splits_path: Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    dev_ratio: float = DEFAULT_DEV_RATIO,
) -> Tuple[List[str], List[str]]:
    expected_dev_count = int(round(len(answers) * dev_ratio))

    if splits_path.exists():
        try:
            data = json.loads(splits_path.read_text(encoding="utf-8"))
            if (
                data.get("schema_version") == SPLIT_SCHEMA_VERSION
                and data.get("answers_list_sha256") == answers_list_sha256
                and data.get("split_seed") == split_seed
                and len(data.get("dev_targets", [])) == expected_dev_count
            ):
                dev_targets = list(data["dev_targets"])
                test_targets = list(data["test_targets"])
                if (
                     _sha256_wordlist(dev_targets) == data.get("dev_targets_sha256")
                    and _sha256_wordlist(test_targets) == data.get("test_targets_sha256")
                    and sorted(dev_targets + test_targets) == answers
                ):
                    return dev_targets, test_targets
        except ( json.JSONDecodeError, KeyError, TypeError):
            pass

    rng = random.Random(split_seed)
    shuffled = list(answers)
    rng.shuffle(shuffled)
    dev_targets = sorted(shuffled[:expected_dev_count])
    test_targets = sorted(shuffled[expected_dev_count:])

    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_seed": split_seed,
        "dev_ratio": dev_ratio,
        "answers_list_sha256": answers_list_sha256,
        "num_answers": len(answers),
        "num_dev_targets": len(dev_targets),
        "num_test_targets": len(test_targets),
        "dev_targets_sha256": _sha256_wordlist(dev_targets),
        "test_targets_sha256": _sha256_wordlist(test_targets),
        "dev_targets": dev_targets,
        "test_targets": test_targets,
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dev_targets, test_targets


def load_dictionary(
    data_dir: Path = DEFAULT_DATA_DIR,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> WordleDictionary:
    """Load and validate the Wordle answer and allowed-guess dictionaries and target splits."""
    answers_path = data_dir / ANSWERS_FILENAME
    allowed_path = data_dir / ALLOWED_FILENAME
    splits_path = data_dir / SPLITS_FILENAME

    if not answers_path.exists():
        raise FileNotFoundError(f"Missing answers file: {answers_path}")
    if not allowed_path.exists():
        raise FileNotFoundError(f"Missing allowed guesses file: {allowed_path}")

    ans_bytes = answers_path.read_bytes()
    all_bytes = allowed_path.read_bytes()

    answers_file_sha256 = _sha256_bytes(ans_bytes)
    allowed_file_sha256 = _sha256_bytes(all_bytes)

    answers = _validate_words(
        ans_bytes.decode("utf-8").splitlines(), ANSWERS_FILENAME
    )
    non_answers = _validate_words(
        all_bytes.decode("utf-8").splitlines(), ALLOWED_FILENAME
    )
    allowed_guesses = sorted(set(answers) | set(non_answers))

    answers_list_sha256 = _sha256_wordlist(answers)
    guesses_list_sha256 = _sha256_wordlist(allowed_guesses)

    answer_to_idx = {w: i for i, w in enumerate(answers)}
    guess_to_idx = {w: i for i, w in enumerate(allowed_guesses)}

    answer_idx_to_guess_idx = np.array(
        [guess_to_idx[w] for w in answers], dtype=np.int32
    )
    guess_idx_to_answer_idx = np.full(len(allowed_guesses), -1, dtype=np.int32)
    guess_idx_to_answer_idx[answer_idx_to_guess_idx] = np.arange(
        len(answers), dtype=np.int32
    )
    is_answer_mask = guess_idx_to_answer_idx >= 0

    dev_targets, test_targets = _load_or_create_splits(
        answers=answers,
        answers_list_sha256=answers_list_sha256,
        splits_path=splits_path,
        split_seed=split_seed,
    )

    return WordleDictionary(
        answers=tuple(answers),
        allowed_guesses=tuple(allowed_guesses),
        answers_file_sha256=answers_file_sha256,
        allowed_file_sha256=allowed_file_sha256,
        answers_list_sha256=answers_list_sha256,
        guesses_list_sha256=guesses_list_sha256,
        answer_to_idx=answer_to_idx,
        guess_to_idx=guess_to_idx,
        answer_idx_to_guess_idx=answer_idx_to_guess_idx,
        guess_idx_to_answer_idx=guess_idx_to_answer_idx,
        is_answer_mask=is_answer_mask,
        dev_targets=tuple(dev_targets),
        test_targets=tuple(test_targets),
        dev_set=set(dev_targets),
        test_set=set(test_targets),
        split_seed=split_seed,
    )


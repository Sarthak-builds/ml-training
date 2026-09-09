"""Text-processing exercises: tokenization, stemming, and lemmatization."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable


URL_PATTERN = r"https?://[^\s]+"
TOKEN_PATTERN = re.compile(
    rf"{URL_PATTERN}|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]"
)
VOWELS = frozenset("aeiou")
WORDNET_POSITIONS = {
    "ADJ": "a",
    "ADV": "r",
    "NOUN": "n",
    "VERB": "v",
}
EVALUATION_CORPUS = (
    ("running", "VERB", "run"),
    ("ran", "VERB", "run"),
    ("better", "ADJ", "good"),
    ("cats", "NOUN", "cat"),
    ("were", "VERB", "be"),
    ("hopping", "VERB", "hop"),
)


def tokenize(text: str) -> list[str]:
    """Split text into words, numbers, punctuation, and complete HTTP(S) URLs."""
    return TOKEN_PATTERN.findall(text)


def contains_vowel(word: str) -> bool:
    """Return whether a word contains a vowel, treating consonant-y as a vowel."""
    return any(character in VOWELS or (character == "y" and index > 0) for index, character in enumerate(word))


def ends_with_double_consonant(word: str) -> bool:
    """Return whether a word ends in two identical consonants."""
    return len(word) >= 2 and word[-1] == word[-2] and word[-1] not in VOWELS


def stem_step_1a(word: str) -> str:
    """Apply Porter stemmer step 1a plural rules."""
    normalized_word = word.lower()
    if normalized_word.endswith("sses"):
        return normalized_word[:-2]
    if normalized_word.endswith("ies"):
        return normalized_word[:-2]
    if normalized_word.endswith("ss"):
        return normalized_word
    if normalized_word.endswith("s") and len(normalized_word) > 1:
        return normalized_word[:-1]
    return normalized_word


def stem_step_1b(word: str) -> str:
    """Remove ``ed`` and ``ing`` suffixes, including the double-consonant rule."""
    normalized_word = word.lower()
    suffix = next((suffix for suffix in ("ing", "ed") if normalized_word.endswith(suffix)), None)
    if suffix is None:
        return normalized_word

    stem = normalized_word[: -len(suffix)]
    if not stem or not contains_vowel(stem):
        return normalized_word
    if ends_with_double_consonant(stem) and stem[-1] not in {"l", "s", "z"}:
        return stem[:-1]
    return stem


def porter_stem(word: str) -> str:
    """Apply the lesson's Porter steps 1a and 1b in sequence."""
    return stem_step_1b(stem_step_1a(word))


def wordnet_lemmatize(word: str, pos: str) -> str | None:
    """Look up a lemma in WordNet, returning ``None`` when it has no entry."""
    try:
        from nltk.corpus import wordnet
    except ImportError:
        return None

    wordnet_pos = WORDNET_POSITIONS.get(pos.upper(), "n")
    try:
        return wordnet.morphy(word.lower(), wordnet_pos)
    except LookupError:
        return None


def lemmatize_with_fallback(word: str, pos: str, lookup: Callable[[str, str], str | None] = wordnet_lemmatize) -> str:
    """Prefer a WordNet lemma and fall back to the local Porter implementation."""
    return lookup(word, pos) or porter_stem(word)


def evaluate_lemmatizers() -> dict[str, float]:
    """Measure exact-match accuracy on the compact tagged evaluation corpus."""
    methods: dict[str, Callable[[str, str], str]] = {
        "porter": lambda word, _pos: porter_stem(word),
        "wordnet": lambda word, pos: wordnet_lemmatize(word, pos) or word.lower(),
        "wordnet_with_porter_fallback": lemmatize_with_fallback,
    }
    return {
        name: sum(method(word, pos) == expected for word, pos, expected in EVALUATION_CORPUS) / len(EVALUATION_CORPUS)
        for name, method in methods.items()
    }


def main() -> None:
    """Print a small demonstration of each exercise implementation."""
    sample_text = "Visit https://example.com today. The hopping cats were running."
    print("Tokens:", tokenize(sample_text))
    print("Stems:", [porter_stem(word) for word in ["caresses", "ponies", "hopping", "watched"]])
    print("Lemmas:", [lemmatize_with_fallback(word, pos) for word, pos in [("running", "VERB"), ("cats", "NOUN")]])


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        for method, accuracy in evaluate_lemmatizers().items():
            print(f"{method}: {accuracy:.0%}")
    main()

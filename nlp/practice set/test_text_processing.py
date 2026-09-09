"""Executable checks for the text-processing exercises."""

from text_processing import lemmatize_with_fallback, porter_stem, stem_step_1b, tokenize


def run_tests() -> None:
    """Run the required examples and representative edge-case checks."""
    assert tokenize("Visit https://example.com today.") == ["Visit", "https://example.com", "today", "."]
    assert tokenize("Don't split contractions!") == ["Don't", "split", "contractions", "!"]
    assert stem_step_1b("hopping") == "hop"
    assert stem_step_1b("running") == "run"
    assert stem_step_1b("bled") == "bled"
    assert porter_stem("caresses") == "caress"
    assert porter_stem("ponies") == "poni"
    assert lemmatize_with_fallback("running", "VERB", lookup=lambda _word, _pos: "run") == "run"
    assert lemmatize_with_fallback("hopping", "VERB", lookup=lambda _word, _pos: None) == "hop"
    print("All text-processing exercise checks passed.")


if __name__ == "__main__":
    run_tests()

# Text Processing Exercises

This practice set implements the three exercises from the linked Text Processing lesson.

## Contents

- `text_processing.py` contains a URL-aware tokenizer, Porter stemmer steps 1a and 1b, and a WordNet-first lemmatizer with a Porter fallback.
- `test_text_processing.py` verifies the required URL and double-consonant examples plus representative edge cases.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe "nlp\practice set\test_text_processing.py"
```

## WordNet evaluation

The optional evaluation in `text_processing.py` compares the fallback lemmatizer with plain WordNet and the local Porter implementation on a small tagged, labelled evaluation corpus. It needs `nltk` with the `wordnet` resource installed locally. The core exercises and tests do not require NLTK.

```powershell
.\.venv\Scripts\python.exe "nlp\practice set\text_processing.py" --evaluate
```

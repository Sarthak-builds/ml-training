# ml-training

A personal collection of machine learning exercises and notes, spanning PyTorch
fundamentals, neural-network training pipelines, classic scikit-learn workflows,
and classical NLP text processing.

## Layout

| Path | Contents |
| --- | --- |
| `tensor.ipynb` | PyTorch tensor creation, indexing, and operations |
| `autograd.ipynb` | Automatic differentiation and the backward pass |
| `nn.ipynb`, `optim.ipynb` | `nn.Module` models and optimizers |
| `dataset_loader.ipynb`, `pipeline.ipynb` | `Dataset`/`DataLoader` and an end-to-end training pipeline |
| `finetune_optimizer.ipynb` | Training and evaluating an MLP on Fashion-MNIST |
| `pytorch/assets/` | Small datasets used by the PyTorch notebooks |
| `perceptron-training/` | Perceptron and customer-churn ANN walkthroughs |
| `nlp/practice set/` | URL-aware tokenizer, Porter stemmer, and WordNet lemmatizer |
| `ml project 1/` | scikit-learn linear-regression pipeline |
| `docs/` | Regression workflow notes |
| `.agents/skills/` | Local agent skill definitions |

## Getting started

Create and activate a virtual environment, then install the packages used by the
notebooks (`torch`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow`, `nltk`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Running the checks

The NLP practice set ships with an executable check script:

```powershell
.\.venv\Scripts\python.exe "nlp\practice set\test_text_processing.py"
```

Run the linear-regression pipeline:

```powershell
.\.venv\Scripts\python.exe "ml project 1\linear_regression.py"
```

The notebooks are intended to be opened and run top to bottom in Jupyter.

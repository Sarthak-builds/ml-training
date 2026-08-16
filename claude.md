# Claude Guidelines for Machine Learning Project

This document outlines instructions and guidelines for Claude / Antigravity when working on the `ml-training` project.

## Project Structure
- `docs/`: Project documentation and guidelines.
- `ml project 1/`: Directory for our first Machine Learning project.
- `.venv/`: Python virtual environment (managed by `uv`).

## Stack & Libraries
- **Language**: Python 3.14+
- **Environment & Dependency Manager**: `uv`
- **Libraries**:
  - `numpy`: Numerical calculations
  - `pandas`: Data manipulation and analysis
  - `scikit-learn`: Machine learning algorithms (to be added)
  - `matplotlib` / `seaborn`: Visualization (to be added)

## Guidelines
1. **Clean Code**: Use functions for logical steps (loading data, preprocessing, training, evaluating).
2. **Reproducibility**: Always set random seeds (e.g., `random_state=42`) for train/test splits and model initialization.
3. **Data Inspection**: Always inspect the data (shape, missing values, data types) before performing preprocessing or modeling.
4. **Documentation**: Document key choices (e.g., feature selection, scaling, model choice) in the `docs/` folder.

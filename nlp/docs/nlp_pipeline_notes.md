# NLP Pipeline Notes

An NLP pipeline turns raw language data into a reliable, maintainable model-powered feature. Each stage should be tracked with versioned data, code, and evaluation results.

## 1. Data acquisition

- Define the task, target users, language coverage, and success criteria.
- Collect data from permitted sources such as public datasets, product logs with consent, or human annotation.
- Record dataset provenance, licences, collection dates, and known coverage gaps.
- Protect personal and sensitive information; remove or mask it before training or analysis.

## 2. Text preparation

- Remove duplicates, corrupted records, markup, and irrelevant boilerplate.
- Standardise text encoding (usually UTF-8), whitespace, punctuation, and language labels where appropriate.
- Split data into training, validation, and test sets without leakage between them.
- Create labels and annotation guidelines; check agreement and class balance.

## 3. Processing

- Tokenise text using a method compatible with the planned model.
- Apply task-specific transformations such as lowercasing, normalisation, stop-word handling, stemming, or lemmatisation only when they help the chosen approach.
- Convert processed text into features or embeddings.
- Build reproducible preprocessing functions and save their configuration with the model.

## 4. Modelling

- Start with a simple baseline such as TF-IDF with logistic regression or a small rule-based system.
- Choose a model architecture that fits the task, available data, latency budget, and hardware limits.
- Train on the training set and tune hyperparameters using the validation set.
- Version the model, training data, code, and experiment settings.

## 5. Evaluation

- Select metrics that match the task: accuracy, precision, recall, F1, ROC-AUC, BLEU, ROUGE, perplexity, or human evaluation.
- Evaluate once on a held-out test set after model choices are final.
- Inspect errors by language, topic, input length, class, and demographic group where appropriate.
- Check robustness, fairness, safety, privacy, and latency in addition to aggregate scores.

## 6. Deployment

- Package preprocessing and the model together so inference matches training.
- Expose the model through a batch job, API, application service, or on-device runtime as appropriate.
- Use staged releases such as shadow, canary, or A/B deployments.
- Provide rollback, authentication, rate limiting, logging, and clear failure behaviour.

## 7. Monitoring

- Monitor service health: latency, error rate, throughput, availability, and resource usage.
- Track input quality, data drift, prediction distributions, and task metrics when feedback becomes available.
- Watch for unsafe outputs, prompt injection attempts, privacy incidents, and bias regressions.
- Set alert thresholds and assign owners for investigation.

## 8. Updating

- Retrain when performance degrades, data changes, new labels arrive, or requirements change.
- Validate every candidate against the current production model and a fixed test set.
- Re-run safety, fairness, privacy, and regression checks before release.
- Record the reason for each update and retain a rollback-ready previous version.

## Suggested practice flow

1. Choose a small classification dataset.
2. Build a reproducible cleaning and tokenisation step.
3. Train a baseline, then compare it with an embedding-based model.
4. Report metrics and analyse a sample of mistakes.
5. Package the best pipeline behind a simple inference interface and define monitoring signals.

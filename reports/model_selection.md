# Model Selection: why Logistic Regression

Eight algorithms were compared in [`notebooks/00_model_comparison.ipynb`](../notebooks/00_model_comparison.ipynb): KNN, Logistic Regression, SVC, Decision Tree, Random Forest, Gradient Boosting (GBM), XGBoost and a Neural Network (MLP).

All models used the same stratified 80/20 split, 3-fold `GridSearchCV` scored on macro recall, and class balancing (`class_weight='balanced'`, or balanced sample weights for GBM, XGBoost and MLP).

## Recall per class on the test set

| Class   | KNN   | LR    | SVC   | DT    | RF    | GBM   | XGBoost | MLP   |
|---------|-------|-------|-------|-------|-------|-------|---------|-------|
| ALLERGY | 0.983 | 0.968 | 0.968 | 0.968 | 0.968 | 0.968 | 0.968   | 0.968 |
| COLD    | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000   | 0.985 |
| COVID   | 0.276 | 0.951 | 0.951 | 0.951 | 0.951 | 0.951 | 0.951   | 0.954 |
| FLU     | 0.945 | 0.905 | 0.905 | 0.905 | 0.905 | 0.905 | 0.905   | 0.905 |

## What the comparison shows

1. **Seven of eight models give the same predictions.** LR, SVC, DT, RF, GBM and XGBoost agree with each other on essentially every test row (prediction agreement = 1.000); MLP differs on about 4 rows (a few COLD cases go to COVID). Accuracy is about 0.93 for all of them.
2. **That is the ceiling of the data, not a bug.** Some symptom patterns appear under more than one illness, and FLU is by far the largest class. Estimated best possible accuracy on this data is about 0.93, and the models reach it.
3. **KNN is the only clear loser.** It has no class weighting, so FLU dominates each neighbourhood vote: recall is 0.40 for COLD and 0.28 for COVID (balanced accuracy about 0.65).
4. **Accuracy, micro-F1 and Hamming loss carry the same information** for multi-class problems, so balanced accuracy, macro-F1 and per-class recall were used to compare models.

## Why Logistic Regression

When the metrics tie, the deciding factors are practical:

- **Simplest model that reaches the ceiling.** Adding boosting or a neural network brings no measurable gain here.
- **Interpretable.** The coefficients show which symptoms drive each prediction (see `reports/figures/coefficients_heatmap.png`), which is much easier to explain than a boosted ensemble.
- **Fast and light.** It trains in seconds, the saved model is a few KB, and there are no extra dependencies such as XGBoost.
- **Gives probabilities.** `predict_proba` outputs a probability per illness, which is more useful than a hard label when classes overlap.
- **Stable.** The tuning score is flat for `C >= 0.1`, so the result does not depend on a lucky hyperparameter.

## Caveats

- The comparison uses a single train/test split.
- The recall table above comes from one run of the comparison notebook; re-running it can shift third-decimal values, especially for MLP.

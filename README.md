# Symptom-Based Illness Classification with Logistic Regression

Predicting COVID, Flu, Cold or Allergy from 20 yes/no symptoms, on a heavily imbalanced dataset. Eight algorithms were compared; **Logistic Regression** was chosen because it matches the best of them and is the simplest and most explainable.

> \*\*Note:\*\* the dataset is synthetic. This project is about handling class imbalance and choosing metrics, not about clinical diagnosis.

## Results

Test set: 8,891 rows (20% stratified split).

|Metric|Test set|5-fold CV (train)|
|-|-|-|
|Accuracy|0.932|0.934 ± 0.004|
|Balanced accuracy|0.956|0.962 ± 0.003|
|Macro F1|0.812|0.817 ± 0.008|

|Class|Precision|Recall|Test rows|
|-|-|-|-|
|ALLERGY|1.00|0.97|3,276|
|COLD|0.49|1.00|205|
|COVID|0.50|0.95|410|
|FLU|1.00|0.90|5,000|

!\[Confusion matrix](reports/figures/confusion\_matrix.png)

## Key Findings

1. **Class weights made the model useful.** COVID and COLD are only 7% of the data. Without weighting, recall was 0.52 for COVID and 0.64 for COLD, even though accuracy was still 0.93. With `class\_weight='balanced'` they rise to 0.95 and 1.00.

   !\[Baseline vs balanced recall](reports/figures/baseline\_vs\_balanced\_recall.png)

2. **Accuracy hides the problem.** The model with and without class weights has the same accuracy (about 0.93). Balanced accuracy (0.77 → 0.96) and per-class recall show the real difference.
3. **The model flags rather than misses.** Recall for COVID and COLD is high, but precision is about 0.5: 392 FLU patients were predicted as COVID and 85 as COLD, because their symptom patterns are identical. In a screening setting this is usually the safer error, but it means about half of COVID flags are false alarms.
4. **Accuracy has a ceiling of about 0.93.** Some symptom patterns appear under several illnesses, so no model can do better with these features. Seven of eight models reached the same predictions.
5. **Coefficients are readable.** Itchy symptoms and pink eye point to Allergy, fever points away from Allergy, runny/stuffy nose and loss of taste/smell point away from COVID, and GI/breathing symptoms point away from Cold.

   !\[Coefficients](reports/figures/coefficients\_heatmap.png)

   ## Why Logistic Regression

   Across KNN, Logistic Regression, SVC, Decision Tree, Random Forest, GBM, XGBoost and a neural network, every model except KNN reached the same accuracy and nearly the same per-class recall. With no performance difference, Logistic Regression wins on simplicity, speed, interpretability and calibrated probabilities. Full comparison: [`reports/model\_selection.md`](reports/model_selection.md).

   ## Dataset

* 44,453 records, 20 binary symptom features, 4 classes (FLU 56%, ALLERGY 37%, COVID 5%, COLD 2%)
* No missing values or duplicates
* Data dictionary: [`data/README.md`](data/README.md)

  ## Approach

1. Check data quality and class distribution.
2. Stratified 80/20 train/test split (`random\_state=10`).
3. Logistic Regression with `class\_weight='balanced'`; tune `C` with 5-fold stratified `GridSearchCV` scored on macro recall.
4. Evaluate with accuracy, balanced accuracy, macro F1, per-class precision/recall and a confusion matrix.
5. Inspect coefficients to see which symptoms drive each prediction.
6. Save the model with `joblib`.

   ## How to Run

   ```bash
git clone https://github.com/resiayukp/symptom-classification-covid-flu-cold.git
cd symptom-classification-covid-flu-cold
pip install -r requirements.txt
jupyter notebook notebooks/01\_logistic\_regression.ipynb
```

   Use the saved model:

   ```python
import joblib
import pandas as pd

model = joblib.load('models/logistic\_regression.joblib')
features = pd.read\_csv('data/raw/covid\_flu\_cold.csv').columns\[:20]

patient = pd.DataFrame(0, index=\[0], columns=features)
patient.loc\[0, \['FEVER', 'NAUSEA', 'VOMITING', 'SHORTNESS\_OF\_BREATH']] = 1

print(model.predict\_proba(patient))   # order: ALLERGY, COLD, COVID, FLU
```

   ## Project Structure

   ```
symptom-classification-covid-flu-cold/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                       # data dictionary and source
│   └── raw/covid\_flu\_cold.csv
├── notebooks/
│   ├── 00\_model\_comparison.ipynb       # 8 algorithms compared
│   └── 01\_logistic\_regression.ipynb    # final model
├── models/
│   └── logistic\_regression.joblib
└── reports/
    ├── model\_selection.md              # why Logistic Regression
    └── figures/                        # charts used in this README
```

   ## Limitations

* Synthetic data: symptoms follow generated rules, so results will not transfer to real patients.
* Overlapping labels cap accuracy at about 0.93 and keep COVID/COLD precision near 0.5.
* Single train/test split for the model comparison; the final model also reports 5-fold CV.

  ## Next Steps

* Choose a probability threshold based on the cost of a missed COVID case versus a false alarm.
* Calibrate predicted probabilities and check them with a reliability curve.
* Repeat the approach on real, public clinical or survey data.

  ## Contact

  **Resiayu K P** — [GitHub](https://github.com/resiayukp) · [LinkedIn](https://linkedin.com/in/resiayu) · [Email](https://mail.google.com/mail/?view=cm&fs=1&to=resiayu.97@gmail.-com)


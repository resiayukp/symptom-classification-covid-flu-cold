# Data

## Source

[covid\_flu\_cold.csv](https://www.kaggle.com/datasets/walterconway/covid-flu-cold-symptoms)

The data is **synthetic**: symptom patterns look generated from rules per illness, not collected from real patients. Results should not be interpreted clinically.

## Shape

* 44,453 rows, 21 columns
* No missing values, no duplicate rows

## Target: `TYPE`

|Class|Rows|Share|
|-|-|-|
|FLU|25,000|56.2%|
|ALLERGY|16,381|36.9%|
|COVID|2,048|4.6%|
|COLD|1,024|2.3%|

## Features

20 binary symptom columns (1 = present, 0 = absent):

COUGH, MUSCLE\_ACHES, TIREDNESS, SORE\_THROAT, RUNNY\_NOSE, STUFFY\_NOSE, FEVER, NAUSEA, VOMITING, DIARRHEA, SHORTNESS\_OF\_BREATH, DIFFICULTY\_BREATHING, LOSS\_OF\_TASTE, LOSS\_OF\_SMELL, ITCHY\_NOSE, ITCHY\_EYES, ITCHY\_MOUTH, ITCHY\_INNER\_EAR, SNEEZING, PINK\_EYE

## Label encoding used in the notebooks

ALLERGY = 0, COLD = 1, COVID = 2, FLU = 3


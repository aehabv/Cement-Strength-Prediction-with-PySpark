# Cement strength prediction
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Spark](https://img.shields.io/badge/PySpark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-0a2c46?style=flat)

How strong will a batch of concrete be, given what went into it? I tried four regression models in PySpark's MLlib to predict compressive strength from the mix ingredients and the age of the sample, then used the best one to answer a few practical questions.

I built this to learn Spark's machine learning library. With 1,030 rows the data is far too small to need Spark, and scikit-learn would do the same job faster.

## The data

The [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) from the UCI repository: 1,030 samples, no missing values.

| Feature | Unit |
|---|---|
| cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate | kg per m³ of mix |
| age | days |
| **csMPa** (target) | compressive strength in MPa |

## What I did

1. **Fixed skew.** Age is heavily right-skewed (skewness 3.26), so I capped it at the 1st and 99th percentiles and took log(age + 1).
2. **Split** the data 70/30 into training and test sets, with a fixed seed.
3. **Compared four models:** linear regression, a decision tree, a random forest and gradient-boosted trees. Each model was tuned with cross-validation over a small parameter grid.
4. **Looked at which features matter,** using the linear model's coefficients and p-values and the tree models' feature importances.
5. **Used the best model** to predict the strength of a given mix, and to find how much cement gives the strongest mix.

## Results

Test-set RMSE (lower is better):

| Model | RMSE (MPa) |
|---|---|
| **Gradient-boosted trees** | **6.77** |
| Linear regression | 7.64 |
| Random forest | 7.79 |
| Decision tree | 8.45 |

The linear model explains 78% of the variance on the test set (R² 0.78), against 83% on the training set, so it doesn't overfit much. Gradient boosting does best, but it only gains about 0.9 MPa on the linear model.

**What matters most:** age, cement and water came out on top in every tree model. In the linear model all eight features are significant at the 5% level. Water is the only one with a negative coefficient, which fits the engineering rule that more water makes weaker concrete.

**Predicting a mix:** for 540 kg/m³ cement, 162 water, 2.5 superplasticizer, 1,040 coarse and 676 fine aggregate at 28 days, gradient boosting predicts 66.99 MPa. The measured strength is 79.99 MPa, so it's 13 MPa short. That's about twice its average error, which makes sense, because this mix is at the very top of the cement range.

**How much cement?** Holding everything else fixed, predicted strength climbs from about 28 MPa at 100 kg/m³ to 67 MPa at 390 kg/m³, then stays flat up to 540. So 390 kg/m³ is the least cement that gets the highest predicted strength. Tree models can't predict beyond the values they've seen, so the flat top is partly a limit of the model rather than a fact about concrete.

## Two bugs I fixed

When I came back to this notebook I found two mistakes in the prediction cells:

- **The age didn't match training.** The model was trained on the transformed age (capped, then log(age + 1)), but new inputs were fed the raw age in days. A prediction for 28 days was really a prediction for an age far outside anything the model had seen. A `prepare_input` helper now applies the same transform to new inputs.
- **The cement search stopped too early.** It stepped cement up by 5 and quit after ten steps without improvement. Gradient-boosted trees predict in flat steps, so it quit straight away and reported the first value it tried, 100 kg/m³. It now scores the whole range and takes the best.

Both fixes changed the answers. The 28-day prediction went from 73.45 to 66.99 MPa. The old figure was closer to the real value, but only by luck. The best cement amount went from 100 to 390 kg/m³.

## Caveats

- **One train/test split.** I report no confidence intervals, and with only about 300 test rows the RMSE differences between the random forest, the linear model and the decision tree are probably not meaningful.
- **The p-values assume a linear model with normal errors.** I didn't check the residuals.
- **Small run-to-run differences.** The random forest and decision tree scores move by a few hundredths between runs. Gradient boosting and the linear model gave the same numbers every time.

## How to run

You need Java 8, 11 or 17 for Spark.

```bash
pip install pyspark pandas jupyter
jupyter notebook Cement_Strength_Prediction.ipynb
```

The cell asking for an age waits for you to type a number. If there's no prompt (for example when the notebook runs headless), it uses 28 days.

The saved outputs come from a run with PySpark 3.5.1. It reproduced the original linear model results exactly (RMSE 7.64529).

## What I'd like to do next

Add a scikit-learn baseline, use bootstrapping to put an interval around each model's RMSE, and check the linear model's residuals.

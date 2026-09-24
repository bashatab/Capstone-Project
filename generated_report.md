# Module 2 Generated Results

Loaded Titanic once through sns.load_dataset and saved titanic.csv immediately.


## Dataset profile

Shape: **891 rows × 15 columns**.

Missing-value percentages:

|             |   Missing % |
|:------------|------------:|
| deck        |   77.2166   |
| age         |   19.8653   |
| embarked    |    0.224467 |
| embark_town |    0.224467 |


## Missing-value handling

- **deck: 77.22% missing.** Missingness was too high for reliable imputation, so 'Missing' was encoded as a separate category.
- **age: 19.87% missing.** Between 5% and 30%, so median imputation was used (28.00).
- **embarked: 0.22% missing.** Under 5%, so rows missing this field were dropped (2 rows at this stage).
- **embark_town: 0.22% missing.** Under 5%, so rows missing this field were dropped (0 rows at this stage).


## Univariate analysis

- **age: 65 IQR outliers** using [2.50, 54.50].
- **fare: 114 IQR outliers** using [-26.76, 65.66].
- Fare mean = **32.10**, median = **14.45**, mode = **8.05**. Because the mean is greater than the median, fare is **right-skewed**.


## Bivariate analysis

### Survival rate by sex

|        |   Survival rate |
|:-------|----------------:|
| female |        0.740385 |
| male   |        0.188908 |

### Survival rate by passenger class

|    |   Survival rate |
|---:|----------------:|
|  1 |        0.626168 |
|  2 |        0.472826 |
|  3 |        0.242363 |

### Survival rate by sex and passenger class

| sex    |   pclass |   survival_rate |
|:-------|---------:|----------------:|
| female |        1 |        0.967391 |
| female |        2 |        0.921053 |
| female |        3 |        0.5      |
| male   |        1 |        0.368852 |
| male   |        2 |        0.157407 |
| male   |        3 |        0.135447 |

The two strongest absolute off-diagonal correlations are **pclass vs fare (-0.548)** and **sibsp vs parch (0.415)**.


## Multivariate data story

1. **Survival by sex:** The chart compares average survival rates across sex categories. The numeric values in the table above provide the exact comparison; the chart makes the gap easier to see.
2. **Class and sex together:** Survival varies across both passenger class and sex, showing that one feature alone does not fully describe the pattern. The combined breakdown is a better multivariate explanation than class-only or sex-only views.
3. **Fare, class and survival:** Fare distributions differ strongly across passenger classes, so fare partly carries class information. The survival split within each class helps assess whether fare still separates outcomes after class is considered.
4. **Age and survival:** The overlapping distributions show that age alone does not perfectly separate survivors and non-survivors. Age can still contribute useful information when combined with sex, class, family size and fare.


## EDA standardization check

|      |   before_mean |   before_std |   after_mean |   after_std |
|:-----|--------------:|-------------:|-------------:|------------:|
| age  |       29.3152 |      12.9776 |  2.71749e-16 |           1 |
| fare |       32.0967 |      49.6695 |  1.39871e-16 |           1 |


## Classification

Class balance before splitting:

|   survived |   Proportion |
|-----------:|-------------:|
|          0 |     0.616162 |
|          1 |     0.383838 |

A stratified split was used so train and test partitions preserve the target proportions. Preprocessing was fitted only through the training pipeline and test data was used only for transformation and evaluation.

| Model               |   Accuracy |   Precision |   Recall |     F1 |    AUC |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.7933 |      0.8636 |   0.5507 | 0.6726 | 0.8292 |
| Random Forest       |     0.8156 |      0.8000 |   0.6957 | 0.7442 | 0.8300 |


## Imbalance handling comparison

| Strategy              |   Precision |   Recall |     F1 |
|:----------------------|------------:|---------:|-------:|
| Baseline              |      0.7931 |   0.6667 | 0.7244 |
| Class weight balanced |      0.7297 |   0.7826 | 0.7552 |
| SMOTE training only   |      0.7397 |   0.7826 | 0.7606 |

**Conclusion:** On this split, **SMOTE training only** produced the highest F1 score. The final choice should consider both recall and precision because a higher recall may come with more false positives.


## Random Forest tuning

Best parameters: `{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 100}`  
Best cross-validation F1: **0.7459**  
OOB score: **0.8272**


## Regression side-task

| Model             |     MAE |    RMSE |     R2 |   Adjusted R2 |
|:------------------|--------:|--------:|-------:|--------------:|
| Linear Regression | 20.8977 | 30.5328 | 0.3975 |        0.3617 |

The residual-variance quartile ratio was **40.83**. Using 2.0 as a transparent diagnostic threshold, the residual plot **shows evidence of non-constant variance**. This is a visual/diagnostic conclusion rather than a formal statistical test.


## Final comparison and recommendation

### Classification metrics

| Model               |   Accuracy |   Precision |   Recall |     F1 |    AUC |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.7933 |      0.8636 |   0.5507 | 0.6726 | 0.8292 |
| Random Forest       |     0.8156 |      0.8000 |   0.6957 | 0.7442 | 0.8300 |

### Regression metrics, separate scale

| Model             |     MAE |    RMSE |     R2 |   Adjusted R2 |
|:------------------|--------:|--------:|-------:|--------------:|
| Linear Regression | 20.8977 | 30.5328 | 0.3975 |        0.3617 |

The recommended classifier is **Random Forest**. On the held-out test set it achieved accuracy **0.8156**, precision **0.8000**, recall **0.6957**, F1 **0.7442**, and AUC **0.8300**. F1 was used as the primary selection measure because it balances precision and recall. The final decision should still consider the operational cost of false positives versus false negatives. The complete preprocessing-plus-model pipeline was saved and successfully reloaded; predictions for three raw rows were `[0, 0, 0]`.

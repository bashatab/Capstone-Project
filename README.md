# Module 2: Analytics Pipeline

This module implements one cohesive Titanic analytics and machine-learning workflow.

## Run

From the repository root:

```bash
python -m pip install -r analytics/requirements.txt
python analytics/module2_analytics.py
```

The first successful run uses `sns.load_dataset("titanic")` and immediately saves `analytics/titanic.csv`. Later runs use only that committed CSV.

## Generated deliverables

- `titanic.csv`: offline fallback
- `generated_report.md`: measured results and written interpretations
- `classification_metrics.csv`
- `imbalance_comparison.csv`
- `regression_metrics.csv`
- `best_classification_pipeline.joblib`
- `charts/`: univariate plots, four data-story charts, correlation heatmap, confusion matrices, ROC curves, decision tree and residual plot

## Design decisions

- EDA missing-value handling follows the assignment thresholds and records exact measured percentages.
- Modeling starts with a stratified split before preprocessing.
- Numeric imputation/scaling and categorical imputation/encoding are inside a `ColumnTransformer` and `Pipeline`, so preprocessing is fitted on training data only.
- SMOTE is inside an imbalanced-learn pipeline and therefore applies only during training.
- The saved artifact is the complete fitted pipeline, not only the estimator.

## Important

Review `generated_report.md` after execution. Use the actual measured values and rewrite the interpretations in your own words before submission.

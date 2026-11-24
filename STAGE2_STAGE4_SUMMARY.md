# Stage 2 and Stage 4 contribution

## Stage 2
- Data cleaning and preparation code
- Training and model evaluation code
- Three candidate model types compared: Ridge, Random Forest,
  and Gradient Boosting
- MLflow tracking and registered model information
- Model artifact and tests

## Stage 4
- PSI and station-type evaluation
- v1 versus v2 retraining and promotion evaluation
- Updated findings files
- Holdout predictions

## Recorded evaluation
- v1 check MAE: 5.8593
- v2 check MAE: 2.5028
- Improvement: 57.28%
- Promotion decision: promote v2
- Holdout predictions: 250 rows

The raw train.csv and later.csv files are intentionally excluded.
The DVC pointer files are included instead.

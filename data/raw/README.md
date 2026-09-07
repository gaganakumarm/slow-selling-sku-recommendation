# M5 Dataset Setup

Obtain the official **M5 Forecasting - Accuracy** competition data from Kaggle and place these files in this directory:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`

These files are needed only for offline training. They are excluded from Git and Docker; do not commit the raw dataset. The API and dashboard can serve the saved portfolio metrics and recommendations without raw CSVs.

From the repository root, after installing dependencies, run `python -m src.pipeline --max-series 500` only when you intend to regenerate the outputs. This replaces existing model, features, metrics, and recommendations. See the root [README](../../README.md) for setup and limitations.

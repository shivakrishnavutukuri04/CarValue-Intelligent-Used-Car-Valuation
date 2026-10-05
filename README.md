# CarValue AI — Unique UI Deployment

A redesigned Streamlit interface for the supplied Used Car Price Prediction model.

## Design direction

This version intentionally moves away from the previous dark/cyan dashboard style.

- Editorial automotive visual language
- Warm paper background
- Graphite/black panels
- Acid-lime accent
- Asymmetric valuation workspace
- Large typography
- Compact specification cards
- Custom CSS vehicle illustration
- Separate Valuation / Methodology / Model Report views
- Responsive layout for smaller screens

## Runtime

The app loads the supplied `model.pkl` directly. The training scripts are retained under `training/` for reference but are not imported by the website.

The supplied model requires scikit-learn 1.9.0, so the deployment requirement is pinned to that version.

## Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Cloud

Upload this folder to GitHub and create a Streamlit app with:

- Main file: `app.py`
- Python dependencies: `requirements.txt`

Do not add MLflow, Optuna or XGBoost to runtime requirements unless the web app itself imports them.

## Important

The price conversion uses the supplied project's fixed reference rate:

`1 USD = ₹86.0`

This is not a live FX rate.

The model performance values shown in the Model Report page come from the supplied project README and are not re-evaluated during deployment.

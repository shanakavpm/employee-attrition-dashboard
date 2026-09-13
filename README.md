# Employee Attrition Dashboard

This project analyses the Saudi Employee Attrition Dataset and presents a Streamlit dashboard for HR decision support. It includes data quality checks, employee attrition visualisations, two prediction models, global feature importance, and group-level fairness diagnostics.

The dashboard also provides an adjustable high-risk threshold. To avoid presenting in-sample predictions as validation evidence, individual risk estimates are shown only for the held-out test records.

## Project pipeline

```text
Original Excel data
    -> cleaning and validation
    -> transparent dashboard fields
    -> train/test split and five-fold cross-validation
    -> Logistic Regression and Random Forest comparison
    -> global feature importance and fairness diagnostics
    -> Streamlit dashboard for HR review
```

## Why these methods

- Logistic Regression is a transparent baseline.
- Random Forest can capture non-linear relationships.
- The selected model is the one with the best hold-out F1 score.
- Class weighting is used instead of synthetic oversampling. This avoids creating artificial employee records.
- Preprocessing is inside each model pipeline, so test-set information cannot affect training.
- Permutation importance measures the effect of shuffling one feature on held-out F1. It indicates association, not causation.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

The supplied original dataset must remain at:

```text
data/raw/Saudi Employee Attrition Dataset/Original Dataset of Employee Attrition.xlsx
```

The raw workbook, cleaned employee-level data, and generated outputs are excluded from Git to avoid publishing record-level data and reproducible build artifacts. Download the dataset from the citation below, place the workbook at the path above, and run `python run_pipeline.py` to regenerate the local files.

## Project structure

```text
app.py                 Streamlit dashboard
run_pipeline.py        Reproducible pipeline runner
src/data.py            Loading, cleaning, validation, and dashboard fields
src/modeling.py        Modelling, evaluation, fairness, and explanations
src/config.py          Paths and shared settings
data/raw/              Supplied original data
data/processed/        Generated clean data
outputs/               Generated evidence tables
```

## Responsible use and limitations

The dataset is survey-based and represents a limited context. It does not establish causal effects or guarantee that results generalise to another organisation. The dashboard is a tool for human review and retention support, not an automated employment decision system.

## Dataset citation

Alqahtani, H., Almagrabi, H. and Alharbi, A. (2025) *Saudi Employee Attrition Dataset*. Mendeley Data, V1. https://doi.org/10.17632/6z2hty8php.1

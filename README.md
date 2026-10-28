# applied-ml-project

This project trains and evaluates a supervised classifier on the UCI Statlog (Vehicle Silhouettes)
dataset: predicting which of four model vehicles (bus, van, Saab, Opel) a 2D silhouette belongs to,
using 18 numeric shape features extracted from the silhouette. A logistic regression baseline is
compared against a random forest ensemble via cross-validation, and the stronger model is evaluated
on a held-out test set.

**Dataset:** [Statlog (Vehicle Silhouettes)](https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes) (Mowforth & Shepherd) — 846 silhouette examples, 18 shape features per row.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the notebook

```bash
source .venv/bin/activate
jupyter notebook modeling.ipynb
```

Run all cells from top to bottom to reproduce preprocessing, model comparison, and evaluation.
`prepare_data.py` combines the raw UCI chunk files into `vehicle.csv`, and `build_notebook.py` is the
script used to generate `modeling.ipynb` from scratch (run it, then
`jupyter nbconvert --to notebook --execute --inplace modeling.ipynb` to re-execute).

## Report

`module_summary.pdf` is the written applied ML report (built from `module_summary.html` via
WeasyPrint), covering the modeling approach, results, non-technical interpretation, limitations/bias,
and references.

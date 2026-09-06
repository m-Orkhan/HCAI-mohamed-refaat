# Human-Centric AI Projects

Mohamed Refaat · Matriculation number 670229

All four projects live in one Django project and are reachable from the home page.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

Then open http://127.0.0.1:8000/

The SQLite database is included in the repository, so no migration step is required.
Tested on Python 3.12 and 3.13 (macOS and Linux).

## Project 1 Supervised Learning Interface

Upload a CSV, explore it, train a model and make predictions.

Expected CSV format: feature columns first, target as the **last column**. The
separator is detected automatically. Choose "group data into categories" for
classification or "predict a number" for regression when uploading.


## Project 2 Explainability

Decision tree and logistic regression on the Palmer Penguins dataset, with a
complexity penalty (λ) slider that selects the model minimising error plus a
complexity term

Also includes counterfactual explanations (the search retries with more samples
and wider variance when none are found, so an unusual target class may take a few
seconds) and PDP and ALE plots.

## Project 3 Active Learning for Learning-to-Defer

The page provides the project report as a PDF download.

The experiment code is in `project3/ml.py`: the TF-IDF baseline classifier, the
simulated expert, the learning-to-defer system, and the comparison of five active
learning query strategies that produced the figures in the report. The interface
itself only serves the report, since the experiments are run offline.

## Project 4 Preference Elicitation

The page provides the report as a PDF download and starts the user study interface.

The study compares pairwise choice against ranking ten movies, using a
Plackett-Luce model over movie features.

## Repository layout

```
project1/ ... project4/    one Django app per project
templates/                 page templates
static/                    stylesheets and the project 3 and 4 reports
report/  report4/          LaTeX sources for the two reports
media/                     generated plots (created at runtime)
```
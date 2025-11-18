# Project 04
#  Title: Anjana – Ensemble Models for Wine Quality Prediction
**Name:** Anjana Dhakal  
**Date:** 11/17/2025  

## Overview
This Jupyter notebook implements ensemble machine learning models to predict wine quality using the Wine Quality dataset. The analysis covers data preparation, feature engineering, training Random Forest and Gradient Boosting models, evaluation using accuracy and weighted F1, and performance comparison with a peer’s (Alissa) models.  
Key goals:
- Explore ensemble modeling approaches for classification.
- Compare performance and generalization of models.
- Reflect on overfitting, feature importance, and model reliability.
- Dataset: Wine Quality (Red) from UCI, 1599 rows, 11 features + target.

## Objective
The goal is to classify wine quality into low, medium, or high categories based on physicochemical properties, and evaluate the effectiveness of different ensemble models.
- Random Forest (100 trees)
- Gradient Boosting (100 estimators)
- Voting Ensemble (Decision Tree + SVM + Neural Network) – peer comparison
  
## Dataset 
Wine Quality Dataset (Red) from UCI: <https://archive.ics.uci.edu/ml/datasets/Wine+Quality>  
- 1599 samples, 12 columns (11 features + target)
- Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- Target: quality (0–10), mapped to 3 classes: low (0), medium (1), high (2)

## Python Libraries for Machine Learning
- `pandas`, `numpy` for data manipulation  
- `matplotlib`, `seaborn` for visualization  
- `scikit-learn` for modeling and evaluation (<https://scikit-learn.org/>) 

## Set Up Machine

Proper setup is critical.
Complete each step in the following guide and verify carefully.

- [SET UP MACHINE](./SET_UP_MACHINE.md)

---

## Set Up Project

After verifying your machine is set up, set up a new Python project by copying this template.
Complete each step in the following guide.

- [SET UP PROJECT](./SET_UP_PROJECT.md)

It includes the critical commands to set up your local environment (and activate it):

```shell
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

**Windows (PowerShell):**

```shell
.\.venv\Scripts\activate
```


## Daily Workflow

Please ensure that the prior steps have been verified before continuing.
When working on a project, we open just that project in VS Code.

### Git Pull from GitHub

Always start with `git pull` to check for any changes made to the GitHub repo.

```shell
git pull
```

### Run Checks 

This mirrors real work where we typically:

1. Update dependencies (for security and compatibility).
2. Clean unused cached packages to free space.
3. Use `git add .` to stage all changes.
4. Run ruff and fix minor issues.
5. Update pre-commit periodically.
6. Run pre-commit quality checks on all code files (**twice if needed**, the first pass may fix things).
7. Run tests.

In VS Code, open your repository, then open a terminal (Terminal / New Terminal) and run the following commands one at a time to check the code.

```shell
git pull
uv sync --extra dev --extra docs --upgrade
uv cache clean
git add .
uvx ruff check --fix
uvx pre-commit autoupdate
uv run pre-commit run --all-files
git add .
uv run pytest
```
## Tools and Libraries

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## Folder Structure 

```shell
notebooks/
  └── project05/
        ├── ensemble-anjana.ipynb
        ├── README.md
        ├── winequality-red.csv

```
## Steps for the project

1. Import and Inspect Data

   - Load Wine Quality dataset (winequality-red.csv)
   - Inspect first rows, shape, summary, missing values

2. Data Exploration and Preparation

   - Map quality to categorical labels (low, medium, high)
   - Map labels to numeric classes (0, 1, 2)
   - Visualize distribution (count plot and histogram)
   - Identify class imbalance (medium dominant ~72%)

3. Feature Selection

   - Input features: all 11 physicochemical properties
   - Target: numeric quality (0–2)

4. Train-Test Split

   - 80% train / 20% test
   - Stratified split to preserve class proportions

5. Train Ensemble Models

   - Random Forest (100 trees)
   - Gradient Boosting (100 estimators)
   - Evaluate using accuracy and weighted F1
   - Confusion matrix for class-wise performance
  

6. Model Evaluation Results  
   
My Test Results

| Model                     | Train Accuracy | Test Accuracy | Accuracy Gap | Train F1  | Test F1   | F1 Gap   |
|---------------------------|----------------|---------------|--------------|-----------|-----------|----------|
| Random Forest (100)       | 1.000000       | 0.887500      | 0.112500     | 1.000000  | 0.866056  | 0.133944 |
| Gradient Boosting (100)   | 0.960125       | 0.856250      | 0.103875     | 0.958410  | 0.841106  | 0.117304 |

Peer Comparison – Alissa’s Test Results

| Model                     | Train Accuracy | Test Accuracy | Train F1  | Test F1   | Acc_Gap  | F1_Gap   |
|---------------------------|----------------|---------------|-----------|-----------|----------|----------|
| Voting (DT + SVM + NN)    | 0.917905       | 0.865625      | 0.900570  | 0.842276  | 0.052280 | 0.058294 |
| Gradient Boosting (100)   | 0.960125       | 0.856250      | 0.958410  | 0.841106  | 0.103875 | 0.117304 |


7.  Final Thoughts & Insights

   - Random Forest achieves highest test accuracy (0.8875) and F1 (0.8661) but overfits (perfect training).
   - Gradient Boosting provides slightly lower accuracy (~0.856) but better generalization (smaller gaps).
   - Alissa’s Voting Ensemble shows the most balanced performance with smallest gaps, demonstrating robustness.
   - Preferred Model: Gradient Boosting for reliability and interpretability.


## Feature Importance (Gradient Boosting)

  - Alcohol – strongest predictor
  - Volatile acidity – taste influence
  - Sulphates – stability
  - Sulfur dioxide – preservation

## Visualizations
   - Confusion matrices show class-wise predictions
   - Bar charts compare test accuracy, F1, and train-test gaps
   - Feature importance highlights top predictors

## Next Steps
   - Apply SMOTE and class weighting to improve minority-class F1
   - Use stacking with XGBoost/LightGBM
   - Engineer feature interactions (e.g., alcohol × sulphates)
   - Use 10-fold cross-validation and track ROC-AUC
   - Add UCI white wine dataset for cross-type validation
 

## Git add-commit-push to GitHub

Anytime we make working changes to code is a good time to git add-commit-push to GitHub.

1. Stage your changes with git add.
2. Commit your changes with a useful message in quotes.
3. Push your work to GitHub.

```shell
git add .
git commit -m "started project04"
git push -u origin main
```




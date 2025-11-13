# Project 03
# Title: Anjana- Titanic Fare Prediction
**Name:** Anjana Dhakal  
**Date:** 11/12/2025  

## Overview
This Jupyter notebook implements regression models to predict passenger fares from the Titanic dataset. It demonstrates data preparation, feature engineering, linear and advanced regression techniques (Ridge, ElasticNet, Polynomial), and model evaluation using metrics like R², RMSE, and MAE. The analysis focuses on limited features (age, family_size, sex) to highlight underfitting and the impact of model complexity.
Key goals:
Explore fare prediction challenges.
Compare model performance on test data.
Reflect on feature importance and overfitting.
Dataset: Titanic (891 rows from Seaborn), target = fare.

## Objective
The goal is to predict passenger fare based on selected features and compare model performance.
Three features:
Case 1(Age) – passenger’s age.
Case 2(Family Size) – number of family members on board.
Case 3(Age + Family Size) – combining age and family size.
Case 4(Sex) – passenger’s gender (encoded).

## Dataset 
Titanic Passenger Dataset (Predict survival based on demographic and travel features)

We use the built-in dataset from Seaborn:
titanic = sns.load_dataset('titanic')

## Python Library for Machine Learning: scikit-learn
We use scikit-learn, built on NumPy, SciPy, and matplotlib
   - Read more at <https://scikit-learn.org/>
   - Scikit-learn supports classification, regression, and clustering.
   - This project applies regression (predict continuous fare values).


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
  └── project04/
        ├── ml04_anjana.ipynb
        ├── README.md
```
## Steps for the project
## 1. Import and Inspect Data
- **Libraries:** `pandas`, `seaborn`, `matplotlib`, `numpy`, `sklearn`  
- **Dataset:** Titanic data from Seaborn  
- **Key steps:**
  - Inspect first 5 rows
  - Check shape (891 rows, 15 cols)
  - Check for missing values
  - Display summary statistics
  

## 2. Data Exploration and Preparation
- **Handle Missing Values:**  
  - Impute age with median 
  - Drop rows with missing fare 
- **Feature Engineering:**  
  - family_size = sibsp + parch + 1  
  - One-hot encode embarked (C, Q, S, missing)  

## 3. Feature Selection
- **Selected features for modeling:**
  - Case 1: `age`  
  - Case 2: `family_size`  
  - Case 3: `age + family_size`  
  - Case 3: `sex_encoded`
- **Target variable:** `fare`  

## 4. Train a Regression Model (Linear Regression)
- Train/test split: 80% train, 20% test (random_state=123)  
- Model trained for each case  
- Evaluation metrics: R², RMSE, MAE  
- Coefficients reported; underfitting observed across cases  

## 5. Alternative Models
- Ridge (α=1.0): Minimal change from linear 
- ElasticNet (α=0.3, l1_ratio=0.5): Slight improvement  
- Polynomial Regression (deg 3): Best performer; visualized for family_size
- Higher degree (deg 8): Overfits extremes


## Summary Table of Model Performance


| Model / Case                     | R² Train | R² Test | RMSE Train | RMSE Test | MAE Train | MAE Test |
| -------------------------------- | -------- | ------- | ---------- | --------- | --------- | -------- |
| **Case 1 – Age**                 | 0.010    | 0.003   | 51.92      | 37.97     | 28.89     | 25.29    |
| **Case 2 – Family Size**         | 0.050    | 0.022   | 50.86      | 37.61     | 27.80     | 25.03    |
| **Case 3 – Age + Family Size**   | 0.073    | 0.050   | 50.23      | 37.08     | 26.61     | 24.28    |
| **Case 4 – Sex**                 | 0.024    | 0.099   | 51.55      | 36.10     | 28.42     | 24.24    |
| **Linear Regression (All)**      | —        | 0.050   | —          | 37.08     | —         | 24.28    |
| **Ridge Regression**             | —        | 0.050   | —          | 37.08     | —         | 24.28    |
| **ElasticNet Regression**        | —        | 0.054   | —          | 36.99     | —         | 24.21    |
| **Polynomial Regression (deg3)** | —        | 0.064   | —          | 36.79     | —         | 23.14    |



## 6. Final Thoughts & Insights
**Insights:**
  - Polynomial Regression (deg 3) performed best
  - Family size most useful feature; age adds minor value
  - All models underfit due to limited features (low R²)
  - Regularization had minimal impact with few features
- **Challenges:**
  - Fare hard to predict without key drivers (e.g., pclass)  
  - Skew/outliers distort linear fits  
  - Non-linearity captured by poly but risks overfit at high degrees 
- **Next Steps:**
  - Add features like pclass, embarked 
  - Hyperparameter tuning (e.g., poly degree) 
  - Cross-validation; log-transform fare for skew

## Bonus: Housing Price Prediction (Continuous Target)

- Dataset: Dataset: Housing.csv (545 rows; features: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus; target: price)
- Models: Models: Linear Regression, Ridge, ElasticNet, Polynomial (deg 3 & 8) tested on feature cases (area only, bedrooms+bathrooms, area+bedrooms+bathrooms+stories, all features)
- Key insights:
  - Linear/Ridge with all features achieved R² ≈ 0.628 (best linear fit)
  - ElasticNet slightly lower (R² 0.609); useful for feature selection
  - Polynomial deg 3 overfit severely (R² -11.3); deg 8 worse (R² -17.6) due to high dimensionality and outliers
  - Area and room counts (bedrooms, bathrooms, stories) most impactful; categorical encodings (e.g., mainroad=yes) boosted performance
  - Skewed price distribution and outliers challenge linear models; log-transform recommended next
  - Generalization improves with more features but risks overfitting without regularization in complex setups

| Model / Case                                  | R² Train | R² Test | RMSE Train | RMSE Test  | MAE Train | MAE Test   |
| --------------------------------------------- | -------- | ------- | ---------- | ---------- | --------- | ---------- |
| Linear Regression Case 1 (Age)                | 0.293    | 0.292   | 0.30       | 0.37       | 0.23      | 0.31       |
| Linear Regression Case 2 (Family Size)        | 0.286    | 0.249   | 0.30       | 0.38       | 0.24      | 0.31       |
| Linear Regression Case 3 (Age + Family Size)  | 0.521    | 0.511   | 0.24       | 0.31       | 0.19      | 0.25       |
| Linear Regression Case 4 (Sex)                | 0.704    | 0.672   | 0.19       | 0.25       | 0.15      | 0.20       |
| Ridge Regression (All features)               | —        | 0.628   | —          | 1104660.26 | —         | 825460.74  |
| ElasticNet Regression (All features)          | —        | 0.609   | —          | 1132199.12 | —         | 820335.02  |
| Polynomial Regression Degree 3 (All features) | —        | -11.321 | —          | 6353432.66 | —         | 1892334.80 |
| Polynomial Regression Degree 8 (All features) | —        | -17.571 | —          | 7799893.78 | —         | 2181973.14 |



**Insights:** Linear models with full features excel for interpretability; polynomials fail on multi-feature sets due to curse of dimensionality.


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




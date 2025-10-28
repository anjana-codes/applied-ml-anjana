# Project 02
# Title: Titanic Dataset Analysis

## Overview
This project analyzes the Titanic dataset to explore relationships between passenger characteristics and survival outcomes. The analysis includes data inspection, visualization, feature engineering, and dataset splitting techniques.

## Objective
The main goal is to understand which features — such as age, class, gender, and fare — most influence survival rates, and to practice data preparation and exploration for machine learning applications.


## Dataset 
Titanic Passenger Dataset (Predict survival based on demographic and travel features)

We use the built-in dataset from Seaborn:
titanic = sns.load_dataset('titanic')

## Python Library for Machine Learning: scikit-learn
We use scikit-learn, built on NumPy, SciPy, and matplotlib
   - Read more at <https://scikit-learn.org/>
   - Scikit-learn supports classification, regression, and clustering.
   - This project applies classification (survived vs. not survived).


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

notebooks/
  └── project02/
        ├── ml02_anjana.ipynb
        ├── README.md

## Steps for the project
1. Import and Inspect Data
- Loaded Titanic dataset
- Checked data info, missing values, and correlations

2. Data Exploration and Preparation
- Created visualizations (scatter plots, histograms, count plots)
- Handled missing values
- Engineered new features like family_size and converted categorical data

3. Feature Selection and Justification
- Selected key features for survival prediction
- Defined input (X) and output (y) variables

4. Splitting the Data
- Compared basic and stratified train-test splits
- Discussed class balance and model performance
  
5. Bonus 
- Apply same process for an additional dataset


## Git add-commit-push to GitHub

Anytime we make working changes to code is a good time to git add-commit-push to GitHub.

1. Stage your changes with git add.
2. Commit your changes with a useful message in quotes.
3. Push your work to GitHub.

```shell
git add .
git commit -m "describe your change in quotes"
git push -u origin main
```




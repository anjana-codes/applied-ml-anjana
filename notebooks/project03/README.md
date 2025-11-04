# Project 03
# Title: Anjana- Titanic Survival Classification
**Name:** Anjana Dhakal  
**Date:** 11/04/2025  

## Overview
This project uses the Titanic dataset to predict passenger survival. Three classifiers—Decision Tree, Support Vector Machine (SVC), and Neural Network (NN)—are trained and evaluated across three feature sets to compare their performance.

## Objective
The goal is to predict passenger survival based on selected features and compare model performance.
Three features: 
- Case 1(Alone) – whether a passenger was traveling alone.
- Case 2(Age) – passenger’s age.
- Case 3(Age + Family Size) – combining age and family members on board.


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

```shell
notebooks/
  └── project03/
        ├── ml03_anjana.ipynb
        ├── README.md
```
## Steps for the project
## 1. Import and Inspect Data
- **Libraries:** `pandas`, `seaborn`, `matplotlib`, `numpy`, `sklearn`  
- **Dataset:** Titanic data from Seaborn  
- **Key steps:**
  - Inspect first 10 rows
  - Check for missing values
  - Display summary statistics
- **Missing values:** `age` (177), `embarked` (2), `deck` (688), `embark_town` (2)

## 2. Data Exploration and Preparation
- **Handle Missing Values:**  
  - Fill missing `age` with median  
  - Fill missing `embark_town` with mode  
- **Feature Engineering:**  
  - `family_size` = `sibsp` + `parch` + 1  
  - Map categorical features (`sex`, `embarked`, `alone`) to numeric  

## 3. Feature Selection
- **Selected features for modeling:**
  - Case 1: `alone`  
  - Case 2: `age`  
  - Case 3: `age + family_size`  
- **Target variable:** `survived`  

## 4. Decision Tree Classifier
- Train/test split: Stratified 80% train, 20% test  
- Model trained for each case  
- Evaluation metrics: Accuracy, Precision, Recall, F1-score  
- Confusion matrices plotted  

## 5. Alternative Models
- **SVC (RBF kernel):** Tested for all three cases (~63% accuracy)  
- **Neural Network (MLP):** Tested for Case 3 (age + family_size), Accuracy 66%  

## Summary Table of Model Performance

| Model                 | Case | Features Used       | Accuracy | Precision | Recall | F1-Score | Notes                             |
|-----------------------|------|------------------|----------|-----------|--------|----------|----------------------------------|
| Decision Tree         | 1    | alone            | 62%      | 0.60      | 0.60   | 0.60     | Test recall for survivors lower |
| Decision Tree         | 2    | age              | 61%      | 0.58      | 0.53   | 0.50     | Recall for class 1 very low     |
| Decision Tree         | 3    | age + family_size| 59%      | 0.57      | 0.55   | 0.54     | Overfitting on training data    |
| SVC (RBF)             | 1    | alone            | 63%      | 0.64      | 0.62   | 0.63     | Test recall for survivors lower |
| SVC (RBF)             | 2    | age              | 63%      | 0.64      | 0.62   | 0.63     | Similar performance to Case 1  |
| SVC (RBF)             | 3    | age + family_size| 63%      | 0.64      | 0.62   | 0.63     | Test recall for survivors lower |
| Neural Network (MLP)  | 3    | age + family_size| 66%      | 0.65      | 0.62   | 0.65     | Best recall for survivors       |

## 6. Final Thoughts & Insights
**Insights:**
  - Neural Network performed best  
  - SVC performance consistent but moderate  
  - Decision Tree prone to overfitting  
  - Adding `age + family_size` slightly improved predictions
- **Challenges:**
  - Class imbalance affected survivor prediction  
  - Limited features  
  - Decision Tree overfit easily  
- **Next Steps:**
  - Hyperparameter tuning  
  - Add more features  
  - Cross-validation and class balancing  


## Bonus: Breast Cancer Dataset Classification

- Dataset: `sklearn.datasets.load_breast_cancer()`
- Models: Decision Tree, SVC, Neural Network
- Key insights:
  - Neural Network achieved ~97% accuracy
  - SVC ~98%, Decision Tree ~94%
  - Models highly effective at detecting malignant cases
  - Scaling and features (texture, radius, smoothness) improved performance
  - Minor overfitting in Decision Tree; SVC and MLP required scaling & tuning

| Model                 | Accuracy | Precision (Malignant) | Recall (Malignant) | Precision (Benign) | Recall (Benign) |
|-----------------------|----------|----------------------|------------------|------------------|----------------|
| Decision Tree         | 0.936    | 0.93                 | 0.89             | 0.94             | 0.96           |
| SVC (RBF)             | 0.977    | 0.97                 | 0.97             | 0.98             | 0.98           |
| Neural Network (MLP)  | 0.971    | 0.94                 | 0.98             | 0.99             | 0.96           |

**Insights:** Neural Network performed best overall. All models accurately detected malignant cases; benign cases were slightly more challenging. Feature scaling and model tuning were critical for optimal performance.


## Git add-commit-push to GitHub

Anytime we make working changes to code is a good time to git add-commit-push to GitHub.

1. Stage your changes with git add.
2. Commit your changes with a useful message in quotes.
3. Push your work to GitHub.

```shell
git add .
git commit -m "started project03"
git push -u origin main
```




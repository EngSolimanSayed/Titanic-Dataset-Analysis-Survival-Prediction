# Titanic Dataset Analysis & Survival Prediction

A data analysis and machine learning project based on the Titanic dataset.

The project focuses on exploring passenger data, handling data quality issues, performing Exploratory Data Analysis (EDA), and building a Logistic Regression model to predict passenger survival.

## Project Overview

The Titanic dataset contains information about passengers such as age, sex, passenger class, fare, family relationships, and embarkation details.

The main objectives of this project are:

* Explore and understand the dataset.
* Identify missing values and duplicated records.
* Clean and prepare the data for analysis.
* Perform Exploratory Data Analysis (EDA).
* Analyze relationships between numerical variables.
* Build a Logistic Regression classification model.
* Evaluate the model using different classification metrics.

## Dataset

The dataset is loaded directly using Seaborn:

```python
sns.load_dataset("titanic")
```

The original dataset contains:

* 891 rows
* 15 columns

Some of the main features include:

* survived
* pclass
* sex
* age
* sibsp
* parch
* fare
* embarked
* class
* who
* adult_male
* deck
* embark_town
* alive
* alone

## Data Cleaning

The project includes several data preparation steps.

### Handling Duplicates

The dataset initially contained 107 duplicated rows.

The duplicated records were removed before continuing with the analysis.

### Handling Missing Values

The `age` column contained missing values, which were filled using the median age.

The `deck` column contained a large number of missing values and was removed from the dataset.

The project also attempts to handle missing values in the `embarked` and `embark_town` columns using their corresponding mappings.

## Exploratory Data Analysis

Several visualizations were created to understand the dataset and explore survival patterns.

The analysis includes:

* Survival distribution
* Survivors by gender
* Survivors by passenger class
* Age distribution
* Fare distribution
* Age distribution by survival
* Fare distribution by passenger class
* Correlation matrix

## Main Observations

The exploratory analysis showed several patterns in the dataset:

* The number of passengers who did not survive was higher than the number who survived.
* Female passengers had more survivors than male passengers.
* Passenger class was related to survival outcomes.
* Most passengers were adults.
* Most ticket fares were relatively low compared with the highest fares in the dataset.

## Machine Learning

A Logistic Regression model was used as a binary classification model.

The target variable was:

`survived`

The dataset was split into:

* 70% Training data
* 30% Testing data

The split used `random_state=42` and stratification.

Categorical features were converted into numerical features using:

`pd.get_dummies(drop_first=True)`

## Model Evaluation

The Logistic Regression model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* AUC

### Results

The model achieved:

| Metric   | Result |
| -------- | -----: |
| Accuracy | 83.47% |
| AUC      |   0.90 |

The classification report showed an overall accuracy of approximately 83% on the test set.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook / Google Colab

## Project Structure

```text
Titanic-Analysis/
│
├── titanic.ipynb
└── README.md
```

## How to Run

1. Clone the repository.

2. Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Open `titanic.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.

4. Run the notebook cells sequentially.

## Project Goal

This project was created as a practical application of Data Analysis and Machine Learning concepts, covering the complete workflow from data exploration and cleaning to model training and evaluation.

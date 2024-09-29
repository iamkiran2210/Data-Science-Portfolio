# Car Analysis and Purchase Prediction

![Machine Learning](https://img.shields.io/badge/Machine_Learning-Classification-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)


## 📚 Project Overview

This project focuses on **Car Analysis** and **Purchase Prediction** using Machine Learning. The goal is to develop a classification model that can predict whether a customer will purchase a car based on various factors such as Gender, age and Annual salary. The project includes data preprocessing, model selection, training, and evaluation using Random Forest
classification algorithm.
### Problem Statement
Given a dataset with customer attributes. The task is to build a classification model that predicts whether a customer will buy a car or not. This can be particularly useful for car dealerships to understand customer behavior and improve sales strategies.

---



## ✨ Features

- **Exploratory Data Analysis (EDA)**: Visualizing patterns and trends in the dataset.
- **Data Preprocessing**: Handling missing values, categorical encoding, and scaling.
- **Model Selection**: Trying different classification models such as Logistic Regression, Decision Trees, Random Forest, and XGBoost.
- **Model Evaluation**: Using accuracy, precision, recall, F1-score, and AUC-ROC for evaluation.
- **Hyperparameter Tuning**: Optimizing the model's performance using techniques like GridSearchCV.
- **Prediction**: Predicting whether a customer will purchase a car based on input features.

---
## 🛠️ Installation and Setup

### Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `jupyter`

### Steps to Install

### 1. Clone the repository:
   ```bash
   git clone https://github.com/iamkiran2210/Data-Science-Portfolio.git
   cd Car-Data
  ```

### 2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv_source
```
### Activating virtual environment
#### On Mac :
```bash
venv/bin/activate
```
#### On Windows :
```bash
venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebook (if applicable):
```bash
jupyter-notebook
```

## 📊 Dataset
The dataset used in this project contains the following columns:

`User ID`: The User ID of the customer

`Gender`: The Gender of the customer

`Age`: The age of the customer.

`Annual Salary`: Annual salary of the customer.

`Purchased`: Target variable indicating if the car was purchased (1) or not (0).

### Dataset Source
If you're using a public dataset, include the link here. For example:

* Dataset is available on Kaggle : https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset

## 🚀 Project Workflow
**1. Data Preprocessing:**

* Handling missing values, if any.
* Encoding categorical features using one-hot encoding or label encoding.
* Feature scaling (Standardization/Normalization) for improving model performance.
**2. Exploratory Data Analysis (EDA):**

* Visualizing key features using histograms, scatter plots, and correlation heatmaps.
* Checking for relationships between features and target variable

**3. Model Building:**

* Training classification models like Logistic Regression, Decision Tree, Random Forest, and XGBoost.
* Splitting the dataset into training and testing sets for unbiased evaluation.
* Using cross-validation for more reliable model assessment.
**4. Model Evaluation:**

* Assessing model performance using metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC Curve.
**5. Hyperparameter Tuning:**

* Fine-tuning model parameters using GridSearchCV or RandomizedSearchCV to optimize performance.

**6. Prediction:**

* Making predictions on new customer data and analyzing model performance on unseen data.
## 📈 Results
After training and evaluating various classification models, the Random Forest Classifier performed the best with the following metrics:

**Accuracy:** 81%

**Precision:** 81%

**Recall:** 81%

**F1-Score:** 81%

**AUC-ROC Score:** 0.86
## 📂 Project Structure
 car-purchase-prediction/
```
├── data/
│   └── car_data.csv          # Dataset
│
├── notebooks/
│   └── eda_&_model.ipynb     # EDA & Model building and evaluation
│
├── README.md                 # Project documentation
└── requirements.txt          # Required dependencies
```
## 📚 Future Enhancements
* **Feature Engineering:** Add more features like customer behavior, car attributes, etc.

* **Deployment:** Deploy the model using Streamlit, Flask or FastAPI and create a web-based interface.

* **Deep Learning:** Experiment with advanced models like neural networks for higher accuracy.

## 📄 License

This project is licensed under the MIT License - [MIT](https://choosealicense.com/licenses/mit/)

## Authors

C Kiran Babu - [@iamkiran2210](https://www.github.com/iamkiran2210)


## 📝 Acknowledgments

* Special thanks to the creators of the dataset.
* Inspired by other machine learning projects.

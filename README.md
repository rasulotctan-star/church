# Customer Churn Prediction

## Overview

This project focuses on predicting customer churn using Machine Learning. Customer churn refers to customers who stop using a company's services. By identifying customers who are likely to churn, businesses can take proactive actions to improve customer retention.

## Dataset

The dataset contains customer information such as:

* Customer demographics
* Account information
* Service subscriptions
* Contract type
* Monthly charges
* Total charges
* Customer tenure
* Churn status (Target Variable)

### Target Variable

* **Churn**

  * 1 = Customer leaves the service
  * 0 = Customer stays with the service

## Project Workflow

### 1. Data Preprocessing

* Handle missing values
* Remove unnecessary columns
* Encode categorical variables
* Scale data if necessary

### 2. Exploratory Data Analysis (EDA)

* Analyze customer behavior
* Visualize churn distribution
* Identify important features affecting churn

### 3. Model Building

The following machine learning algorithms can be used:

* Logistic Regression
* Random Forest
* XGBoost
* Decision Tree
* Gradient Boosting

### 4. Model Evaluation

Performance metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Train the model:

```bash
python train.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv
│
├── model/
│   └── model.pkl
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## Business Impact

Customer churn prediction helps organizations:

* Reduce customer loss
* Improve customer satisfaction
* Increase revenue
* Optimize marketing campaigns
* Build targeted retention strategies

## Future Improvements

* Hyperparameter tuning
* Feature engineering
* Model deployment on cloud platforms
* Real-time prediction system

## Author
Rasul Ibrahimov
Developed as a Machine Learning project for customer retention analysis and churn prediction.

import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "shivang"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ShivShivShiv"
mlflow.set_tracking_uri("http://127.0.0.1:5000")


mlflow.set_experiment("Github Integration Experiment")
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
)


X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

mlflow.sklearn.autolog(disable=True)
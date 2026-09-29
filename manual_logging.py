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

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "solver": "lbfgs",
    "max_iter": 100,
    "random_state": 8555,
}




with mlflow.start_run():
  
    mlflow.log_params(params)

    
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)


    model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model",registered_model_name="sk-learn-logistic-reg-model")

    y_pred = lr.predict(X_test)
    metrics = {
    "test_accuracy": accuracy_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred, average='weighted'),
    "test_recall": recall_score(y_test, y_pred, average='weighted'),
    "test_f1": f1_score(y_test, y_pred, average='weighted'),
    }

    mlflow.log_metrics(metrics)

    mlflow.set_tag("Training Info", "Basic LR model for iris data")
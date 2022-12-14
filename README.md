# Prediction of survival using classification models and deploying the best model as a REST API

- Used Titanic dataset from **Kaggle** to build and compare a **variety of machine learning classifiers** with the **sckit-learn** (logistic regression, decision trees, support vector machine, random forest, voting classifier), in order to predict survival of passengers on the Titanic
- Comprised **all phases of machine learning workflow** (e.g., train-test-splitting the data, data exploration, feature engineering (incl. pipelines), optimization of hyperparameters, evaluation via cross-validation)
- Deployed the best classifier (voting classifier) as a **REST API** using **FastAPI**

<img src="./figures/overview_ml_pm.png" height="500" />
Figure : Main perfromance metrics for implemented machine learning classifiers

## Usage
- Run REST API with: 

    ``` cd codes```

    ```uvicorn api:app --reload``` 
- To access interactive API documentation and interface, open ``` http://127.0.0.1:8000/docs```  after starting the API locally.


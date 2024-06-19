import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import fbeta_score, precision_score, recall_score

def get_model_pipeline(model, cat_var, num_var):
    
    if isinstance(model, RandomForestClassifier):
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=1000)
    elif isinstance(model, LogisticRegression):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # categorical feature preprocessor
    cat_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        encoder
    )

    # numerical feature preprocessor
    num_preproc = StandardScaler()

    # features preprocessor
    feature_preproc = ColumnTransformer([        
        ('categorical', cat_preproc, cat_var),
        ('numerical', num_preproc, num_var)
    ],
        remainder='drop'
    )

    # model pipeline
    model_pipe = Pipeline([
        ('features_preprocessor', feature_preproc),
        ('model', model)
    ])

    return model_pipe


def compute_metrics(y_true, y_pred):

    f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)

    return f1, precision, recall


def inference_model(model, X):
    
    preds = model.predict(X)
    return preds


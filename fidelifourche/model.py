import numpy as np
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import pickle
import os
import time
import glob

from fidelifourche.params import LOCAL_REGISTRY_PATH

def train_model(preproc,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray) :

    pipe = make_pipeline(XGBClassifier())

    # Hyperparameter Grid
    grid = {'xgbclassifier__learning_rate': [0.2, 0.3, 0.4, 0.5],
            'xgbclassifier__max_depth' : [4,5,6,7,8],
            'xgbclassifier__gamma':[0,1,2]}

    # Instanciate Grid Search
    search = GridSearchCV(pipe,
                          grid,
                          scoring = 'precision',
                           cv = 5,
                           n_jobs=-1)

    search.fit(X_train,y_train)

    # Params
    params = search.best_params_

    # Estimator = model
    model = search.best_estimator_

    # Score model
    metrics = model.score(X_val,y_val)

    print(f"\n✅ model trained ({len(X_train)} rows)")

    return model, params, metrics

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

def model_rfr(X_train, y_train, X_test, y_test):
    model_rfr = RandomForestRegressor()

    model_rfr.fit(X_train, y_train)
    y_pred = model_rfr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**(1/2)
    train_score = model_rfr.score(X_test, y_test)
    cv_mean = cross_val_score(model_rfr, X_train, y_train, cv = 5).mean()
    baseline_train = 'Baseline train score was 0.63'
    baseline_cv_mean = 'Baseline cv score mean was 0.54'

    return rmse, train_score, cv_mean, baseline_train, baseline_cv_meanda

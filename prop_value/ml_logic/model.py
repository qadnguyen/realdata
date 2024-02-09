from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import pickle

import warnings
warnings.filterwarnings('ignore')

def model_rfr(X_train, y_train, X_test, y_test, X_all, y_all):

    model_rfr = RandomForestRegressor()

    model_rfr.fit(X_train, y_train)
    y_pred = model_rfr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**(1/2)

    train_score = model_rfr.score(X_test, y_test)
    test_score = model_rfr.score(X_test, y_test)
    cv_mean = cross_val_score(model_rfr, X_all, y_all, cv = 5).mean()

    baseline_train = 'Baseline train score was 0.63'
    baseline_cv_mean = 'Baseline cv score mean was 0.54'

    results = {'RMSE': rmse,
               'Train score': train_score,
               'Test score': test_score,
               'CV mean': cv_mean,
               'Baseline train score': baseline_train,
               'Baseline cv score': baseline_cv_mean}
    return results

def model_xgb(X_train, y_train, X_test, y_test, X_all, y_all):

    import pickle

    model_xgb = XGBRegressor()

    model_xgb.fit(X_train, y_train)
    with open('xgb_model.pkl', 'wb') as file:
        pickle.dump(model_xgb, file)

    y_pred = model_xgb.predict(X_test)

    mse = mean_squared_error(y_pred, y_test)
    rmse = mse**(1/2)

    train_score = model_xgb.score(X_train, y_train)
    test_score = model_xgb.score(X_test, y_test)
    cv_mean = cross_val_score(model_xgb, X_all, y_all, cv = 5).mean()

    baseline_train = 'Baseline train score was 0.63'
    baseline_cv_mean = 'Baseline cv score mean was 0.54'

    results = {'RMSE': rmse,
                'Train score': train_score,
                'Test score': test_score,
                'CV mean': cv_mean,
                'Baseline train score': baseline_train,
                'Baseline cv score': baseline_cv_mean}
    return results

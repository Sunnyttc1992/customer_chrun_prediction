import optuna as optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X,y):
    """
    Docstring for tune_model
    
    :param X: Description
    :param y: Description
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**param)
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)
    return study.best_params

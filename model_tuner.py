import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from config import Config

class BaseModelTuner:
    def __init__(self, X_train, y_train, cv_strategy, model_list: list):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv_strategy
        self.model_list = model_list
        self.results_df = None
        self.best_params_dict = None
        self.random_state = Config.RANDOM_STATE



    def tune_models(self, refit_metric='accuracy'):
        pre_tuning_results = []
        n_iterations = Config.RANDOM_SEARCH_ITER_BASE
    
        print(f"Starting randomized search for {len(self.model_list)} models ({n_iterations} iterations each)...")
    
        for model in self.model_list:
            name = model.get_name()
            print(f"Tuning {name}...")
    
            param_distributions = model.get_param_grid()
    
            random_search = RandomizedSearchCV(
                estimator=model.get_estimator(),
                param_distributions=param_distributions,
                n_iter=n_iterations,
                cv=self.cv,
                scoring=['accuracy', 'roc_auc', 'neg_log_loss'],
                refit=refit_metric,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
    
            random_search.fit(self.X_train, self.y_train)
    
            # Best estimator after RandomizedSearchCV
            best_model = random_search.best_estimator_
    
            # Generate predictions on full training set for reporting
            y_pred_proba = best_model.predict_proba(self.X_train)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
    
            # Compute all key metrics
            accuracy = accuracy_score(self.y_train, y_pred)
            roc_auc = roc_auc_score(self.y_train, y_pred_proba)
            ll = log_loss(self.y_train, y_pred_proba)
    
            best_params_raw = random_search.best_params_
            best_params_cleaned = {key.replace('model__', ''): value for key, value in best_params_raw.items()}
    
            pre_tuning_results.append({
                'model_name': name,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'log_loss': ll,
                'best_params': best_params_cleaned
            })
    
            print(f"Tuning for {name} complete.")
            print(f"  Refit Metric ({refit_metric}): {random_search.best_score_:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  LogLoss: {ll:.4f}")
            print(f"  Best Parameters found: {best_params_cleaned}\n")
    
        print("Base models tuning completed.")
        return pre_tuning_results

    def get_results(self):
        """
        Returns the tuning results and best parameters.
        """
        if self.results_df is None or self.best_params_dict is None:
            raise RuntimeError("You must run .tune_models() before getting results.")
        return self.results_df, self.best_params_dict

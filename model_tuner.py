import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from config import Config

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

class BaseModelTuner:
    def __init__(self, X_train, y_train, cv_strategy, model_list: list):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv_strategy
        self.model_list = model_list
        self.results_df = None
        self.best_params_dict = None
        self.random_state = Config.RANDOM_STATE

def tune_models(self, refit_metric='roc_auc'): # it's best to refit on roc_auc or neg_log_loss instead of accuracy
        
        pre_tuning_results = []
        best_params_dict = {} 
        
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
                refit=refit_metric, # This determines random_search.best_score_
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
    
            random_search.fit(self.X_train, self.y_train)
    
            
            best_index = random_search.best_index_

            cv_roc_auc = random_search.cv_results_['mean_test_roc_auc'][best_index]
            cv_accuracy = random_search.cv_results_['mean_test_accuracy'][best_index]
            cv_log_loss = -1 * random_search.cv_results_['mean_test_neg_log_loss'][best_index]
            
            # This is the score for the chosen refit metric
            best_cv_score = random_search.best_score_
            
            best_params_raw = random_search.best_params_
            best_params_cleaned = {key.replace('model__', ''): value for key, value in best_params_raw.items()}
            
            best_params_dict[name] = best_params_cleaned

            pre_tuning_results.append({
                'model_name': name,
                'best_roc_auc_cv': cv_roc_auc,
                'accuracy_cv': cv_accuracy,
                'log_loss_cv': cv_log_loss,
                'best_params': best_params_cleaned
            })
            
            print(f"Tuning for {name} complete.")
            print(f"  Best CV Refit Score ({refit_metric}): {best_cv_score:.4f}")
            print(f"  Mean CV ROC-AUC: {cv_roc_auc:.4f}")
            print(f"  Mean CV Accuracy: {cv_accuracy:.4f}")
            print(f"  Mean CV LogLoss: {cv_log_loss:.4f}")
            print(f"  Best Parameters found: {best_params_cleaned}\n")
                        
        print("Base models tuning completed.")
        
        return pre_tuning_results, best_params_dict

    def get_results(self):
        """
        Returns the tuning results and best parameters.
        """
        if self.results_df is None or self.best_params_dict is None:
            raise RuntimeError("You must run .tune_models() before getting results.")
        return self.results_df, self.best_params_dict

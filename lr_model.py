from sklearn.linear_model import LogisticRegression
from base_model import TunableBaseModel

class LogisticRegressionModel(TunableBaseModel):
    
    def get_name(self) -> str:
        return 'lr'

    def _create_model_object(self) -> object:
        return LogisticRegression(
            random_state=self.random_state, 
            max_iter=self.max_iter
        )

    def get_param_grid(self) -> dict:
        param_grid = [
            {'model__penalty': ['l1'], 
             'model__C': loguniform(0.01, 100), 
             'model__solver': ['liblinear'], 
             'model__n_jobs': [1]
            },
            {'model__penalty': ['l2'], 
             'model__C': loguniform(0.01, 100), 
             'model__solver': ['lbfgs'], 
             'model__n_jobs':[-1]
            }
        ]
        return param_grid

    def get_tuned_blueprint(self, params: dict) -> object:
        # This is for the stacker
        params.pop('n_jobs', None) #Remove n_jobs or it will conflict with the other n_jobs
        
        return LogisticRegression(
            **params,  # This are the best parameters 
            random_state=self.random_state, 
            max_iter=self.max_iter,
            n_jobs=1 
        )

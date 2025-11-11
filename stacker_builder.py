from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from typing import List
from scipy.stats import loguniform, randint
from base_model import TunableBaseModel 
from config import Config



class StackingModelBuilder:
    """
    Assembles a StackingClassifier by leveraging each model's
    own `get_estimator` pipeline.
    """

    def __init__(self, 
                 model_list: List[TunableBaseModel], 
                 base_model_params: dict, 
                 cv_strategy):
        
        self.model_list = model_list
        self.base_params = base_model_params
        self.cv = cv_strategy
        self.final_pipeline = self._build_pipeline()
        self.param_distributions = self._build_param_distributions()

    def _build_pipeline(self) -> StackingClassifier: # <-- Returns the stacker directly
        
        estimators_l0_tuned = []
        for model in self.model_list:
            name = model.get_name()
            if name not in self.base_params:
                print(f"Warning: No parameters found for model '{name}'. Skipping.")
                continue
            
            estimator_pipeline = model.get_estimator()
            

            tuned_params_cleaned = self.base_params[name].copy()
            
            tuned_params_cleaned.pop('n_jobs', None)
            
            tuned_params_prefixed = {
                f'model__{key}': value 
                for key, value in tuned_params_cleaned.items()
            }
            
            estimator_pipeline.set_params(**tuned_params_prefixed)
            
            estimators_l0_tuned.append((name, estimator_pipeline))

        print(f"Stacker built with models: {[name for name, _ in estimators_l0_tuned]}")

        #Define Meta-Model Pipeline
        meta_model_l1 = Pipeline([
            ('model_meta', LogisticRegression(
                random_state=Config.RANDOM_STATE, 
                max_iter=Config.MAX_ITER,
                solver='liblinear'
            ))
        ])

        # 3. Create Stacker
        # This IS the final model
        stacker = StackingClassifier(
            estimators=estimators_l0_tuned,
            final_estimator=meta_model_l1,
            cv=self.cv,
            n_jobs=-1, # Use -1 for parallel execution of CV folds
            passthrough=False 
        )
        
        return stacker

    def _build_param_distributions(self) -> dict:
        """
        Private method to define the meta-model's tuning distributions.
        The 'model__' prefix is removed as the stacker is no longer
        wrapped in an outer pipeline.
        """
        param_distributions_stack_final = {
            'final_estimator__model_meta__C': loguniform(0.01, 100.0),
            'final_estimator__model_meta__penalty': ['l1', 'l2']
        }
        return param_distributions_stack_final

    def get_pipeline(self) -> StackingClassifier: # <-- Returns StackingClassifier
        return self.final_pipeline

    def get_param_grid(self) -> dict:
        """
        Returns the parameter distributions for tuning the meta-model.
        """
        return self.param_distributions

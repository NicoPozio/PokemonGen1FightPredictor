from sklearn.ensemble import RandomForestClassifier
from base_model import TunableBaseModel

class RandomForestModel(TunableBaseModel):

    def get_name(self) -> str:
        return 'rf'

    def _create_model_object(self) -> object:
        return RandomForestClassifier(
            criterion='gini', 
            random_state=self.random_state, 
            n_jobs=-1
        )

    def get_param_grid(self) -> dict:
        param_grid = {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': randint(10, 30),
            'model__max_features': ['sqrt'],
            'model__min_samples_leaf': randint(2, 10)
        }
        return param_grid

    def get_tuned_blueprint(self, params: dict) -> object:
        # This is for the stacker
        return RandomForestClassifier(
            **params, 
            random_state=self.random_state, 
            n_jobs=1 
        )


    def get_estimator(self) -> Pipeline:
        """
        Builds the pipeline for the Decision Tree.
        This model is scale-invariant, so NO scaler is added.
        """
        core_model = self._create_model_object()
        self.pipeline = Pipeline([
            ('model', core_model)
        ])
        return self.pipeline

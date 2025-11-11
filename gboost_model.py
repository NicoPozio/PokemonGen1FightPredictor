from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform, loguniform
from base_model import TunableBaseModel


class GradientBoostingModel(TunableBaseModel):

    def get_name(self) -> str:
        return 'gb'

    def _create_model_object(self) -> object:
        """
        Create the base Gradient Boosting model.
        """
        return GradientBoostingClassifier(
            random_state=self.random_state
        )

    def get_param_grid(self) -> dict:
        """
        Parameter distribution for randomized search.
        Designed for ~10 iterations under 10k samples and 58 features.
        """
        param_grid = {
            'model__n_estimators': randint(100, 400),
            'model__learning_rate': uniform(0.01, 0.2),   # range: 0.01–0.21
            'model__max_depth': randint(2, 6),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__subsample': loguniform(0.7, 0.3),        # range: 0.7–1.0
            'model__max_features': ['sqrt', 'log2', None]
        }
        return param_grid

    def get_tuned_blueprint(self, params: dict) -> object:
        """
        Used when building the tuned stacker model.
        GradientBoosting does not support n_jobs, so we omit it.
        """
        return GradientBoostingClassifier(
            **params,
            random_state=self.random_state
        )

    def get_estimator(self) -> Pipeline:
        """
        Builds the pipeline for Gradient Boosting.
        No scaler needed (tree-based model).
        """
        core_model = self._create_model_object()
        self.pipeline = Pipeline([
            ('model', core_model)
        ])
        return self.pipeline

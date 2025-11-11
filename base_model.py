from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class TunableBaseModel(ABC):
    """
    Abstract base class (interface) for a tunable model.
    It defines the contract that all concrete model classes must follow.
    """
    def __init__(self, random_state, max_iter=None):
        self.random_state = random_state
        self.max_iter = max_iter
        self.pipeline = None 

    @abstractmethod
    def get_name(self) -> str:
        """Returns the short name (key) of the model."""
        pass

    @abstractmethod
    def _create_model_object(self) -> object:
        """
        Protected method for creating the core model instance.
        This is what the concrete class implements.
        """
        pass

    @abstractmethod
    def get_param_grid(self) -> dict:
        """Returns the parameter grid for GridSearchCV."""
        pass


    @abstractmethod
    def get_tuned_blueprint(self, params: dict) -> object:
        """
        Returns a new, unfitted model instance configured with
        the best parameters and fixed settings for stacking.
        
        Args:
            params (dict): A dictionary of the best hyperparameters
                           (already cleaned, without 'model__' prefix).
        """
        pass
    
    def get_estimator(self) -> Pipeline:
        """
        This method is now shared by all models.
        It builds the standard pipeline.
        """
        core_model = self._create_model_object()
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', core_model)
        ])
        return self.pipeline


from sklearn.neighbors import KNeighborsClassifier
from base_model import TunableBaseModel

class KNNModel(TunableBaseModel):
    
    def get_name(self) -> str:
        return 'knn'

    def _create_model_object(self) -> object:
        return KNeighborsClassifier(
            n_jobs=1 
        )

    def get_param_grid(self) -> dict:
        param_grid = {
            'model__n_neighbors': randint(3, 30),
            'model__weights': ['distance']
        }
        return param_grid

    def get_tuned_blueprint(self, params: dict) -> object:
        # This is for the stacker
        return KNeighborsClassifier(
            **params, 
            n_jobs=1, 
        )

    

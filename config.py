import os

class Config:
    #Project constants
    RANDOM_STATE = 42
    MAX_TURNS = 30 
    ID_COLUMN_NAME = 'battle_id'
    TARGET_COLUMN_NAME = 'player_won'
    
    #Project file paths
    COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
    DATA_PATH = os.path.join('../input', COMPETITION_NAME)
    TRAIN_FILE = os.path.join(DATA_PATH, 'train.jsonl')
    TEST_FILE = os.path.join(DATA_PATH, 'test.jsonl')

    #Modeling parameters
    CV_SPLITS_BASE = 10
    CV_SPLITS_META = 5
    RANDOM_SEARCH_ITER_BASE = 10
    RANDOM_SEARCH_ITER_META = 10
    MAX_ITER = 2000
    
    #Hyperparameter grid
    C_VALUES_TO_TEST = [0.01, 0.1, 1, 10, 100]

    #Feature/Target names
    TARGET_COLUMN = 'player_won'
    ID_COLUMN = 'battle_id'

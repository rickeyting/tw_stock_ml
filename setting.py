import os


BASE_DIR = os.getcwd()


TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'training_data/')

TEST_DATA_PATH = os.path.join(BASE_DIR, 'test_data/')

PREDICTING_DATA_PATH = os.path.join(BASE_DIR, 'pred_data/')

SHARE_DATA_PATH = os.path.join(BASE_DIR, 'base_data.csv')

TRANSFORMER_PATH = os.path.join(BASE_DIR, 'models', 'transformer')
os.makedirs(TRANSFORMER_PATH, exist_ok=True)
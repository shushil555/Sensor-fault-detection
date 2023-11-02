import os
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME
"""
Defining common constant variable that are required in training pipeline
"""
SAVED_MODEL_DIR = os.path.join('saved_models')

TARGET_COLUMN = 'class'
PIPELINE_NAME:str = 'sensor'
ARTIFACT_DIR:str = 'artifact'
FILE_NAME:str = 'sensor.csv'


TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME:str = 'test.csv'

PREPROCESSING_OBJECT_FILE_NAME = 'preprocessing.pkl'
MODEL_FILE_NAME = 'model.pkl'
SCHEMA_FILE_PATH = os.path.join('config','schema.yaml')
SCHEMA_DROP_COLS = 'drop_columns'

"""
Data Ingestion Related Constant Start with Data_Ingestion Variable Name
"""

DATA_INGESTION_COLLECTION_NAME:str = 'car'
DATA_INGESTION_DIR_NAME:str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR:str = 'feature_store'
DATA_INGESTION_INGESTED_DIR :str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2


"""
Data Validation Related Constant Start with DATA_VALIDATION"

"""
DATA_VALIDATION_DIR:str = 'data_validation'
DATA_VALIDATION_VALID_DIR:str = 'validated'
DATA_VALIDATION_INVALID_DIR:str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT:str = 'drfift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str ='report.yaml'


"""
DATA  TRANSFORMATION RELATED CONSTANT
"""
DATA_TRANSFORMATION_DIR_NAME:str ='data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str= 'transformed_object'


"""
Model Trainer Related Constant
"""

MODEL_TRAINER_DIR_NAME:str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR:str ='trained.model'
MODEL_TRAINER_TRAINED_MODEL_NAME:str = 'model.pkl'
MODEL_TRAINER_EXPECTED_SCORE:float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05


"""
Model Evaluation Related COntant
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUTION_DIRECTORY_NAME:str = 'model_evalution'
MODEL_EVALUTION_REPORT_FILE_NAME:str = 'report.yaml'


"""
Model Pusher Related constant
"""

MODEL_PUSHER_DIR_NAME:str = 'model_pusher'
MODEL_PUSHER_SAVED_MODEL_SIR= SAVED_MODEL_DIR


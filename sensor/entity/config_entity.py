
from dataclasses import dataclass
from datetime import datetime
import os,sys
from sensor.constant.training_pipeline import PIPELINE_NAME,ARTIFACT_DIR
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.constant import training_pipeline


class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        timestamp = timestamp.strftime('%m_%d_%Y_%H_%M_%S')
        self.timestamp = timestamp
        self.pipeline_name:str = training_pipeline.PIPELINE_NAME
        self.artifact_dir :str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME)

        self.feature_store_file_path:str = os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME

        )

        self.training_file_path:str = os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TRAIN_FILE_NAME
        )

        self.testing_file_path:str = os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio:float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name:str = training_pipeline.DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR)
        self.valid_data_dir:str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir:str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path:str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_train_file_path:str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path:str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_test_file_path:str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.data_drift_report_dir:str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT)
        self.drift_report_file_path:str = os.path.join(self.data_drift_report_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.data_transformation_transformed_data_dir = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
        self.transfomed_train_file_path = os.path.join(self.data_transformation_transformed_data_dir,training_pipeline.TRAIN_FILE_NAME.replace('csv','np'))
        self.transfomed_test_file_path = os.path.join(self.data_transformation_transformed_data_dir,training_pipeline.TEST_FILE_NAME.replace('csv','np'))
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir:str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy:float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        

class ModelEvalutionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_evalutaion_dir:str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_EVALUTION_DIRECTORY_NAME)
        self.report_file_path = os.path.join(self.model_evalutaion_dir, training_pipeline.MODEL_EVALUTION_REPORT_FILE_NAME)
        self.changed_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.MODEL_PUSHER_DIR_NAME
        )
        self.model_file_path = os.path.join(self.model_evaluation_dir)
        timestamp = round(datetime.now().timestamp())

        self.saved_model_path = os.path.join(
            training_pipeline.SAVED_MODEL_DIR,
            f"{timestamp}",
            training_pipeline.MODEL_FILE_NAME
        )
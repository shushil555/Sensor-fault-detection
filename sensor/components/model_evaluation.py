from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import ModelTrainerArtifact,ModelEvalutionArtifact,DataValidationArtifact
from sensor.entity.config_entity import ModelEvalutionConfig
import os,sys
import pandas as pd
import numpy as np
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.ml.model.estimator import TargetMapping

from sensor.ml.metrics.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object,load_object,write_yaml_file
from sensor.ml.model.estimator import ModelResolver

class ModelEvaluation:
    def __init__(self, model_eval_config:ModelEvalutionConfig,
                 data_validation_artifact:DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact
                 ) -> ModelEvalutionArtifact:
        try:
            self.model_eval_config=model_eval_config
            self.data_validation_artifact=data_validation_artifact
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)

    def initiate_model_evaluation(self,):
        try:
            logging.info('reading valid train file path and test file path')
            valid_train_file_path =self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            logging.info('coverting into dataframe')
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)
            logging.info('Merging dataframe')
            df = pd.concat([train_df,test_df])
            df = df.replace('na', np.nan)
            df = df.drop_duplicates()
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetMapping().to_dict(),inplace=True)
            logging.info('dropping target columns')
            df.drop(TARGET_COLUMN,axis=1,inplace=True)

            
            model_resolver = ModelResolver()
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            is_model_accepted = True

            if not model_resolver.is_model_exist():
                logging.info('no better model better found')
                model_evaluation_artifact = ModelEvalutionArtifact(is_model_accepted=is_model_accepted,
                    changed_accuracy=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact, 
                    best_model_metric_artifact=None)
                return model_evaluation_artifact
        
            logging.info('models are there')
            latest_model_path = model_resolver.get_best_model_path()
            logging.info('loading latest_model')
            latest_model = load_object(latest_model_path)
            logging.info('loading trained model')
            train_model = load_object(train_model_file_path)


            y_trained_pred = train_model.predict(df)
            y_latest_pred  =latest_model.predict(df)
            logging.info('metrics')
            trained_metric =get_classification_score(y_true,y_trained_pred)
            logging.info(f"trained metrics: {trained_metric}")
            latest_metric = get_classification_score(y_true,y_latest_pred)
            logging.info(f"The latest best model")

            improved_accuracy = trained_metric.f1_score-latest_metric.f1_score

            if self.model_eval_config.changed_threshold < improved_accuracy:
                is_model_accepted=True
            else:
                is_model_accepted=False


            model_evaluation_artifact = ModelEvalutionArtifact(
                is_model_accepted=is_model_accepted,
                changed_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric, 
                best_model_metric_artifact=latest_metric
            )
            model_eval_report = model_evaluation_artifact.__dict__


            write_yaml_file(self.model_eval_config.report_file_path,model_eval_report)
            logging.info(f"Model evaluation artifact {model_evaluation_artifact}")
            return model_evaluation_artifact

            
        except Exception as e:
            raise SensorException(e,sys)
        





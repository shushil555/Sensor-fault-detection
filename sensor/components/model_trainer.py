from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
import os,sys
from xgboost import XGBClassifier
from sensor.utils.main_utils import load_numpy_array_data,load_object, save_object
from sensor.ml.metrics.classification_metric import  get_classification_score
from sensor.ml.model.estimator import SensorModel


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig, data_transformatin_artifact:DataTransformationArtifact) -> None:
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformatin_artifact=data_transformatin_artifact
        except Exception as e:
            raise SensorException(e,sys)
        
    
    def train_model(self, x_train,y_train):
        try:
            xgb = XGBClassifier()
            xgb.fit(x_train,y_train)
            return xgb
        except Exception as e:
            raise SensorException(e,sys)


    def initiate_model_trainer(self, )->ModelTrainerArtifact:
        try:
            test_file_path = self.data_transformatin_artifact.transfomed_test_file_path
            train_file_path = self.data_transformatin_artifact.transfomed_train_file_path
            logging.info('loading the test and train data')
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            classifcation_train_metrics = get_classification_score(y_true=y_train,y_predicted=y_train_pred)
            y_test_pred = model.predict(x_test)
            classifcation_test_metrics = get_classification_score(y_true=y_test,y_predicted=y_test_pred)
            
            logging.info('Overfitting -- Underfitting')
            if classifcation_train_metrics.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception('Model Underfitting')

            logging.info(f'{"*"*50}')
            logging.info('Overfitting Checking')
            diff = abs(classifcation_train_metrics.f1_score-classifcation_test_metrics.f1_score)

            if diff>0.05:
                raise Exception('Model Overfitting')
            
            preprocessor = load_object(file_path=self.data_transformatin_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path,sensor_model)

            #model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classifcation_train_metrics,
                test_metric_artifact=classifcation_test_metrics
            )
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e,sys)
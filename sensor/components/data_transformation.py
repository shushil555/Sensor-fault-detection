from sensor.exception import SensorException
from sensor.logger import  logging
from sensor.entity.config_entity import DataTransformationConfig
from sensor.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
import os,sys
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.utils.main_utils import save_numpy_array_data,save_object
from sensor.ml.model.estimator import TargetMapping
import numpy as np

class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig, data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact 
        except Exception as e:
            raise  SensorException(e,sys)
        
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            logging.info('reading csv')
            data = pd.read_csv(file_path)
            data = data.drop_duplicates()
            return data
        except Exception as e:
            raise SensorException(e,sys)
        

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            logging.info('getting into pipeline')
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    (
                'Imputer', simple_imputer
                    ),
                    (
                'RobustScaler',robust_scaler
                    )
                ]
            )
            return preprocessor

        except Exception as e:
            raise SensorException(e,sys)

    def  initiate_data_transformation(self,):
        try:
            test_data_frame = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            train_data_frame = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            preprocessor = self.get_data_transformer_object()


            #training dataframe
            logging.info('working on training dataframe')
            input_feature_train_df = train_data_frame.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_data_frame[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(TargetMapping().to_dict())


            #testing dataframe
            logging.info('working on testing data')
            input_feature_test_df = test_data_frame.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_data_frame[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(TargetMapping().to_dict())         


            #handling nullvalues
            input_feature_train_df = input_feature_train_df.replace('na', np.nan)
            input_feature_test_df = input_feature_test_df.replace('na', np.nan)



            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

            logging.info('oversampling')
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[ input_feature_test_final, np.array(target_feature_test_final) ]

            save_numpy_array_data(self.data_transformation_config.transfomed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transfomed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)


            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transfomed_train_file_path= self.data_transformation_config.transfomed_train_file_path,
                transfomed_test_file_path=self.data_transformation_config.transfomed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )


            return data_transformation_artifact


        except Exception as e:
            raise SensorException(e,sys)




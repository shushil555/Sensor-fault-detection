from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import ModelPusherArtifact,ModelEvalutionArtifact
from sensor.entity.config_entity import ModelPusherConfig
import os,sys
import shutil

class ModelPusher:
    def __init__(self,model_evaluation_artifact:ModelEvalutionArtifact, model_pusher_config:ModelPusherConfig):
        try:
            self.model_evaluation_artifact=model_evaluation_artifact
            self.model_pusher_config=model_pusher_config
        except Exception as e:
            raise SensorException(e,sys)

    

    def initate_model_pusher(self, )->ModelPusherArtifact:
        try:
            logging.info('initiating model pusher')
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            model_file_path = self.model_pusher_config.model_file_path
            logging.info('making dir')
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            logging.info('makking copy')
            shutil.copy(src=trained_model_path,dst=model_file_path)

            #saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            ##preparing artifact
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path,
                model_file_path=model_file_path
            )
            logging.info(f"model pusher artifac: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e,sys)

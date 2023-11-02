from sensor.entity.artifact_entity import ClassificationMetricsArtifact
from sensor.logger import logging
from sensor.exception import SensorException
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import os,sys


def get_classification_score(y_true, y_predicted)->ClassificationMetricsArtifact:
    try:
        logging.info('into the metrics')
        model_f1_score = f1_score(y_true, y_predicted)
        logging.info(f'model f1_socre: {model_f1_score}')
        model_recall_score = recall_score(y_true,y_predicted)
        logging.info(f'model recall score: {model_recall_score}')
        model_precision_score = precision_score(y_true,y_predicted)
        logging.info(f'model precision score: {model_precision_score}')

        return ClassificationMetricsArtifact(f1_score=model_f1_score,precision_score=model_precision_score,recall_score=model_recall_score)
    except Exception as e:
        raise SensorException(e,sys)
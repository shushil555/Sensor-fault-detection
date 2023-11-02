from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    validation_satus:bool
    valid_train_file_path:str
    invalid_train_file_path:str
    valid_test_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str



@dataclass
class DataTransformationArtifact:
    transfomed_train_file_path:bool
    transfomed_test_file_path:bool
    transformed_object_file_path:bool



@dataclass
class ClassificationMetricsArtifact:
    f1_score:float
    precision_score:float
    recall_score:float



@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    train_metric_artifact:ClassificationMetricsArtifact
    test_metric_artifact:ClassificationMetricsArtifact

@dataclass
class ModelEvalutionArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    best_model_path:str
    trained_model_path:str
    train_model_metric_artifact: ClassificationMetricsArtifact
    best_model_metric_artifact: ClassificationMetricsArtifact


@dataclass
class ModelPusherArtifact:
    saved_model_path:str
    model_file_path:str

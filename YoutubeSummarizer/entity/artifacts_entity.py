from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_ingestion_artifact:str

@dataclass
class DataTransformArtifact:
    data_transform_artifact:str

@dataclass
class ModelTrainingArtifact:
    model_training_artifact:str
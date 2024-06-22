import os,sys
from YoutubeSummarizer.components.data_ingestion import DataIngestion
from YoutubeSummarizer.components.data_transformation import DataTransform


from YoutubeSummarizer.entity.artifacts_entity import DataIngestionArtifact,DataTransformArtifact
from YoutubeSummarizer.entity.config_entity import DataIngestionConfig,DataTransformConfig

from YoutubeSummarizer.logger import logging
from YoutubeSummarizer.exception import CustomException


class TrainingPipeline():
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transform_config = DataTransformConfig()

    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
            logging.info("Unzipping the data from the original file tar.bgz file")

            data_ingestion = DataIngestion()
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Unzipping the data from the original file tar.bgz file is completed")
            logging.info("Data Ingestion is completed")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact)-> DataTransformArtifact:
        try:
            logging.info("Entered the start_data_transformation method of TrainingPipeline class")
            logging.info("Starting the data transformation process")

            data_transform = DataTransform(data_transform_config=self.data_transform_config,
                                           data_ingestion_artifact=data_ingestion_artifact)
            data_transform_artifact = data_transform.Creating_dataset_dict()

            logging.info("Data transformation is completed")
            return data_transform_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
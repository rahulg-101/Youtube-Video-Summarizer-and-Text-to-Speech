import os
import sys
import zipfile
# import ntpath
# import gdown
from YoutubeSummarizer.logger import logging
from YoutubeSummarizer.exception import CustomException
from YoutubeSummarizer.entity.artifacts_entity import DataIngestionArtifact
from YoutubeSummarizer.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self,data_ingestion_config= DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
        

    def extract_zip_file(self,zip_file_path):
        """
        zip_file_path : str
        Extracts the zip file into data directory
        Function returns none
        """
        try:
            data_ingestion_dir = os.path.join(self.data_ingestion_config.data_ingestion_dir)
            os.makedirs(data_ingestion_dir, exist_ok=True)
            os.system('tar -xzvf tmp_en_sum_4407145a-6de3-418f-95a1-f82ff2c470ee.tar.bz2 -C'+data_ingestion_dir)
            
            logging.info(f"Extracted zip file {zip_file_path} into {data_ingestion_dir}")

            return data_ingestion_dir

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self,zip_file_path)-> DataIngestionArtifact:
        """
        This function will return downloaded and unzipped data path
        for our artifacts_entity.py file's DataIngestionArtifact class
        """
        logging.info(f"Entered initiate_data_ingestion method of DataIngestion class")

        try:
            feature_store_path = self.extract_zip_file(zip_file_path)

            data_ingestion_artifact = DataIngestionArtifact(
                data_zip_file_path =  feature_store_path
            )

            logging.info("Exited initiate_data_ingestion method of DataIngestion classs"
            )
            logging.info(f"Data ingestion artifact : {data_ingestion_artifact}")

            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    print(obj.initiate_data_ingestion(r'C:\Users\rahul gupta\Documents\Learning\Projects\Youtube Text Summarizer\Youtube-Video-Summarizer-and-Text-to-Speech\tmp_en_sum_4407145a-6de3-418f-95a1-f82ff2c470ee.tar.bz2'))

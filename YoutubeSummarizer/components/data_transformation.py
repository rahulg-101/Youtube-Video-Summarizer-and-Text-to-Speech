# Require dataconfigartifact in class self method
import os,sys
from YoutubeSummarizer.entity.config_entity import DataTransformConfig
from YoutubeSummarizer.entity.artifacts_entity import DataIngestionArtifact

from YoutubeSummarizer.logger import logging
from YoutubeSummarizer.exception import CustomException
from YoutubeSummarizer.utils.main_utils import *

from datasets import Dataset
from datasets import DatasetDict


class DataTransform:
    def __init__(self,
                 data_transform_config = DataTransformConfig(),
                 data_ingestion_artifact = DataIngestionArtifact):
        self.data_transform_config = data_transform_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def data_conversion_from_text_to_dataframe(self):
        try:
            logging.info("Starting to transform data from textual type to pandas DataFrame")

            train_trans = os.path.join(self.data_ingestion_artifact.data_ingestion_artifact,self.data_transform_config.train_trans)
            train_summ = os.path.join(self.data_ingestion_artifact.data_ingestion_artifact,self.data_transform_config.train_summ)
            print(train_trans)
            val_trans = os.path.join(self.data_ingestion_artifact.data_ingestion_artifact,self.data_transform_config.val_trans)
            val_summ = os.path.join(self.data_ingestion_artifact.data_ingestion_artifact,self.data_transform_config.val_summ)

            test_trans = os.path.join(self.data_ingestion_artifact.data_ingestion_artifact,self.data_transform_config.test_trans)
            test_summ = os.path.join(self.data_ingestion_artifact.data_ingestion_artifact,self.data_transform_config.test_summ)
                                    
            train = extract_pandas_df(train_trans,train_summ)
            val = extract_pandas_df(val_trans,val_summ)
            test = extract_pandas_df(test_trans,test_summ)

            logging.info("Successfully transformed data from textual type to pandas DataFrame")
            
            return train,val,test

        except Exception as e:
            raise CustomException(e,sys)
        
    def converting_to_Dataset_type(self):
        try:
            
            logging.info("Starting to transform data from pandas Dataframe to Dataset type")
            train,val,test = self.data_conversion_from_text_to_dataframe()

            dataset_train = Dataset.from_pandas(train,split='train')
            dataset_val = Dataset.from_pandas(val,split='val')
            dataset_test = Dataset.from_pandas(test,split='test')

            logging.info("cleaning text in the transcripts and summary features")

            ds_train = clean_dataset(dataset_train)
            ds_val = clean_dataset(dataset_val)
            ds_test = clean_dataset(dataset_test)

            return ds_train, ds_val, ds_test
        except Exception as e:
            raise CustomException(e,sys)
        
    def Creating_dataset_dict(self):
        try:
            logging.info("Entered Create_dataset_dict method of DataTransform class")
            logging.info("creating dataset dictionary object suitable with HuggingFace interface")

            ds_train, ds_val, ds_test = self.converting_to_Dataset_type()
            dataset_dict = {
            'train' : ds_train,
            'test': ds_test,
            'validation': ds_val
            }

            data = DatasetDict(dataset_dict)

            data_transform_dir = self.data_transform_config.data_transform_dir
            data.save_to_disk(data_transform_dir)
            logging.info("Exiting Create_dataset_dict method of DataTransform class")

            return data_transform_dir
                
        except Exception as e: 
            raise CustomException(e,sys)


            


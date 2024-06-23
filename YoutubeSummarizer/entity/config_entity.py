import os
from dataclasses import dataclass
from datetime import datetime
from YoutubeSummarizer.constant.training_pipeline import *

@dataclass
class TrainingPipelineConfig:
    artifacts_dir:str = ARTIFACTS_DIR

training_pipeline_config = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str = os.path.join(training_pipeline_config.artifacts_dir,DATA_INGESTION_DIR_NAME)

    data_file_name = DATA_FILE_NAME

@dataclass
class DataTransformConfig:
    data_transform_dir = os.path.join(training_pipeline_config.artifacts_dir,DATA_TRANSFORM_DIR)

    train_trans = TRAIN_TRANSCRIPT
    train_summ = TRAIN_SUMMARY
    test_trans = TEST_TRANSCRIPT
    test_summ = TEST_SUMMARY
    val_trans =VAL_TRANSCRIPT
    val_summ = VAL_SUMMARY

@dataclass
class ModelTrainingConfig:
    model_trainer_dir = os.path.join(training_pipeline_config.artifacts_dir,MODEL_TRAINER_DIR_NAME)

    model_checkpoint = MODEL_TRAINER_CHECKPOINT
    max_input_length = MODEL_TRAINER_MAX_INPUT_LENGTH
    max_target_length = MODEL_TRAINER_MAX_TARGET_LENGTH

    batch_size = MODEL_TRAINER_BATCH_SIZE
    epochs = MODEL_TRAINER_EPOCH
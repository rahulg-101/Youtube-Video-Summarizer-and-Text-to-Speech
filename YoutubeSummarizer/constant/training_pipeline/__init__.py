ARTIFACTS_DIR = "artifacts"

"""
Data Ingestion related constant start with DATA_INGESTION Var Name,
this is useful when you want to change your directories or URL from which you are receiving the files.

"""

DATA_INGESTION_DIR_NAME = "data_ingestion"      # Store unzipped data in this directory

DATA_FILE_NAME = 'tmp_en_sum_4407145a-6de3-418f-95a1-f82ff2c470ee.tar.bz2'

"""
Data Transformation related constant start with DATA_TRANSFORMATION Var Name
"""
DATA_TRANSFORM_DIR = "data_transformation"

TRAIN_TRANSCRIPT = 'text\sum_train\\tran.tok.txt'
TRAIN_SUMMARY = 'text\sum_train\desc.tok.txt'

TEST_TRANSCRIPT = 'text\sum_devtest\\tran.tok.txt'
TEST_SUMMARY = 'text\sum_devtest\desc.tok.txt'

VAL_TRANSCRIPT = 'text\sum_cv\\tran.tok.txt'
VAL_SUMMARY = 'text\sum_cv\desc.tok.txt'


"""

Model trainer related consstant start with MODEL_TRAINER VAR Name
"""

MODEL_TRAINER_DIR_NAME = "model_trainer"    # Create model_trainer folder

MODEL_TRAINER_CHECKPOINT = "facebook/bart-large-cnn"

MODEL_TRAINER_MAX_INPUT_LENGTH = 512
MODEL_TRAINER_MAX_TARGET_LENGTH = 30

MODEL_TRAINER_BATCH_SIZE = 4
MODEL_TRAINER_EPOCH = 5

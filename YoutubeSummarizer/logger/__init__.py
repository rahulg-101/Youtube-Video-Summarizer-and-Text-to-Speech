import logging
import os
from datetime import datetime
from from_root import from_root

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"    # name of the log subdirectory made of current timestamp value

log_path = os.path.join(from_root(), 'log',LOG_FILE)    # Creating a path for 2 sub-directories called "log" > "LOG_FILE" 

os.makedirs(log_path,exist_ok=True)     # Creating the directory using path defined above 

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)     # Creating path for the log-file that will actually reside in the subdirectories created in step 2

logging.basicConfig(filename=LOG_FILE_PATH,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

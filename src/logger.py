#Whenever some execution occurs it should log all the execution in some files so that we will be able to track

import logging #The logging module in Python is used to track events, debug information, warnings, and errors that happen during program execution. It's a better and more flexible alternative to using print() statements.
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True) #Even though there is a file there is a folder keep on updating it

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  #%(asctime)s: Time the log was recorded
    level=logging.INFO
)


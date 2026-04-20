import logging
import os
import re
import sys
from src.loger import get_logger
from src.costum_expection import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.componnent.data_transformation import DataTransformation
from src.componnent.data_transformation import DataTransformationConfig
from src.componnent.model_trainner import ModelTrainer
from src.componnent.model_trainner import ModelTrainerConfig

logger=get_logger(__name__)


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logger.info("Data Ingestion method starts")
        try:
            # load the dataset as dataframe
            df=pd.read_csv("notebook/data/data.csv")
            logger.info("reading the dataset as dataframe")
            logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            #-------------------------------------------------------------------------------------
            logger.info("train test split initiated")
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info("train test split completed")
            logger.info("Data Ingestion method completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error("Error occured in Data Ingestion method")
            raise CustomException(f"error occurred in data ingestion method: {str(e)}", sys)

if __name__=="__main__":
    obj=DataIngestion()
    train,test=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train,test)
    model_trainer=ModelTrainer()
    r2_square=model_trainer.initiate_model_trainer(train_arr,test_arr)
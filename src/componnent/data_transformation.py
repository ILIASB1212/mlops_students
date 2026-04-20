import logging
import os
import re
import sys
from src.loger import get_logger
from src.costum_expection import CustomException
from src.utils import save_object
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

logger=get_logger(__name__)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course"]
            num_pipeline=Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())]
            )
            logger.info(f"numerical columns scalling completed")
            cat_pipeline=Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(drop="first")),
                    ("scaler",StandardScaler(with_mean=False))]
            )
            logger.info(f"categorical columns one-hot encoding completed")
            preprossessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprossessor
        except Exception as e:
            logger.error("Error occured in Data Transformation method ")
            raise CustomException(f"error occurred in data transformation method: {str(e)}", sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            preprossessor_obj=self.get_data_transformation_object()
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logger.info("read training and testing data completed")
            #-------------------------------------------------------------------------------------
            target_column_name="math_score"
            #-------------------------------------------------------------------------------------
            training_feature=train_df.drop(columns=[target_column_name],axis=1)
            training_target=train_df[target_column_name]
            #-------------------------------------------------------------------------------------
            testing_feature=test_df.drop(columns=[target_column_name],axis=1)
            testing_target=test_df[target_column_name]
            logger.info("seperating input and target feature completed")
            logger.info("applying preprocessing object on training and testing features")
            
            input_feature_train_arr=preprossessor_obj.fit_transform(training_feature)

            input_feature_test_arr=preprossessor_obj.transform(testing_feature)
            

            train_arr = np.c_[input_feature_train_arr, np.array(training_target)]

            test_arr = np.c_[input_feature_test_arr, np.array(testing_target)]

            logger.info(f"Saved preprocessing object.")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprossessor_obj
            )

            return(     
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                )
            
        except Exception as e:
            logger.error("Error occured in Data Transformation method ")
            raise CustomException(f"error occurred in data transformation method: {str(e)}", sys)
from pydantic import BaseModel
from requests import request

from src.costum_expection import CustomException
from src.loger import get_logger
import sys
import pandas as pd
import os
from src.utils import load_object





class CustomData(BaseModel):
        gender: str
        race_ethnicity: str
        parental_level_of_education: str
        lunch: str
        test_preparation_course: str
        reading_score: int
        writing_score: int


class PredictPipeline:
    def __init__(self,request:CustomData):
        self.request=request
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            "gender": [self.request.gender],
            "race_ethnicity": [self.request.race_ethnicity],
            "parental_level_of_education": [self.request.parental_level_of_education],
            "lunch": [self.request.lunch],
            "test_preparation_course": [self.request.test_preparation_course],
            "reading_score": [ self.request.reading_score],
            "writing_score": [self.request.writing_score],} 

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
import sys 
import os 
import pandas as pd 
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path)
            processor = load_object(preprocessor_path)
            data_scaled = processor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
    


class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity:str,
            parental_level_of_education:str,
            lunch:str,
            test_preparation_course:str,
            math_score:int,
            reading_score:int,
            writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_dataframe(self):
        try:
            custom_data_input = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],  
                "math score": [self.math_score],  
                "reading score": [self.reading_score],  
                "writing score": [self.writing_score] 
            }

            return pd.DataFrame(custom_data_input)
        

        except Exception as e:
            raise CustomException(e,sys)


        
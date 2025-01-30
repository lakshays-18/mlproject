import sys
import os
import numpy as np
import pandas as pd 
from dataclasses import dataclass 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()
    
    def get_transformer_obj(self):
        try:
            numerical_columns = ['math score','reading score','writing score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )   

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehot",OneHotEncoder()),
                ]
            )


            logging.info("Scaling of numerical features completed")
            logging.info("Encoding of categorical features completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Train and test data reading completed")
            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_transformer_obj()

            target_column_name = "average"

            input_features_train = train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train = train_data[target_column_name]

            input_features_test = test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test = test_data[target_column_name]

            logging.info("Applying preprocessor object on train and test dataframes")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_test)

            train_arr = np.c_[
                input_features_train_arr,np.array(target_feature_train)
            ]
            test_arr = np.c_[
                input_features_test_arr,np.array(target_feature_test)
            ]

            logging.info('Saved preprocessing object')

            save_object(
                file_path = self.data_transformer_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)



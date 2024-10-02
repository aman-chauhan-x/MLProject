import sys
import pandas as pd 
import numpy as np 
import os
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os
from src.mlproject.utiles import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_column = ['reading_score','writing_score']
            categorical_column = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            numerical_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])
            
            categorical_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'categorical columns :{categorical_column}')
            logging.info(f'numerical columns : {numerical_column}')

            preprocessor=ColumnTransformer(
                [
                    ('numerical_pipeline',numerical_pipeline,numerical_column),
                    ('categorical_pipeline',categorical_pipeline,categorical_column)
                ]
            )

            return preprocessor
            

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('reading the train and test file')

            preprocessing_obj=self.get_data_transformer_obj()

            target_col_name="math_score"
            numerical_col_name=["writing_score","reading_score"]

            ##devide the train dataset into independent and dependent fetures

            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train_df[target_col_name]

             ##devide the test dataset into independent and dependent fetures

            input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df=test_df[target_col_name]

            logging.info("applying preprocessing on training and testing dataset")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("saving processed object")

            save_object(

                file_path=self.data_transfromation_config.preprocessor_obj_file_path,obj=preprocessing_obj
            )
            
            return(

                train_arr,test_arr,
                self.data_transfromation_config.preprocessor_obj_file_path

            )

        
        except Exception as e:
            raise CustomException(e,sys)

    
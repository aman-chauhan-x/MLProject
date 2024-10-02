from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_ingestion import DataIngestion
import pandas as pd
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation

from src.mlproject.components.model_trainer import ModelTrainer


if __name__=='__main__':
    logging.info("the execution has started")


    try:
        data_ingestion=DataIngestion()
        train_datapath,test_datapath=data_ingestion.initiate_data_ingestion()
        #data_ingestion_config=DataIngestionConfig()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_a,test_a,_=data_transformation.initiate_data_transformation(train_datapath,test_datapath)

        # Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_a,test_a))



    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)
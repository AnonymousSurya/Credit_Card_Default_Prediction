import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir=os.path.join(artifact_folder)
    transformed_train_file_path=os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path=os.path.join(artifact_dir, 'test.npy') 
    transformed_object_file_path=os.path.join( artifact_dir, 'preprocessor.pkl' )




class DataTransformation:
    def __init__(self,
                 feature_store_file_path):
       
        self.feature_store_file_path = feature_store_file_path

        self.data_transformation_config = DataTransformationConfig()
        
        self.utils =  MainUtils()
    
    
    @staticmethod
    def get_data(feature_store_file_path:str) -> pd.DataFrame:
        """
        Method Name :   get_data
        Description :   This method reads all the validated raw data from the feature_store_file_path and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        """
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"default payment next month": TARGET_COLUMN}, inplace=True)


            return data
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'LIMIT_BAL',
                'EDUCATION',
                'AGE',
                'PAY_0',
                'PAY_2',
                'PAY_3',
                'PAY_4',
                'PAY_5',
                'PAY_6',
                'BILL_AMT1',
                'BILL_AMT2',
                'BILL_AMT3',
                'BILL_AMT4',
                'BILL_AMT5',
                'BILL_AMT6',
                'PAY_AMT1',
                'PAY_AMT2',
                'PAY_AMT3',
                'PAY_AMT4',
                'PAY_AMT5',
                'PAY_AMT6'
            ]
            categorical_columns = [
                "SEX",
                "MARRIAGE"
            ]

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",RobustScaler(with_centering=False))
                ]
            )

            cat_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",RobustScaler(with_centering=False))
                ]
            )

            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

             
    def initiate_data_transformation(self) :
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
            
        """

        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )

        try:
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
           
            
            
            X = dataframe.drop(columns= TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN]
            
            
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 )



            preprocessor = self.get_data_transformer_object()

            X_train_scaled =  preprocessor.fit_transform(X_train)
            X_test_scaled  =  preprocessor.transform(X_test)

            


            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok= True)
            self.utils.save_object( file_path= preprocessor_path,
                        obj= preprocessor)

            train_arr = np.c_[X_train_scaled, np.array(y_train) ]
            test_arr = np.c_[ X_test_scaled, np.array(y_test) ]

            np.save(self.data_transformation_config.transformed_train_file_path,train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path,test_arr)



            return (train_arr, test_arr, preprocessor_path)
        

        except Exception as e:
            raise CustomException(e, sys)
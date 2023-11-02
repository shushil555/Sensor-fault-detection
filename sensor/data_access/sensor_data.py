import sys
from typing import Optional
import  numpy as np
import pandas as pd
import json
from sensor.configuration.mongodb_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME
from sensor.exception import SensorException
from sensor.logger import logging


class SensorData:

    """
    This class helps to export entire mongodb record to as pandas dataframe
    """

    def __init__(self,):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise SensorException(e,sys)
        

    def save_csv_file(self, file_path, collection_name:str, database_name:Optional[str]=None):
        try:
            data_frame = pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json().values()))

            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
                collection.insert_many(records)

            return len(records)
        except Exception as e:
            raise SensorException(e,sys)
        

    def export_collection_as_dataframe(self, collection_name:str, database_name:Optional[str]=None)->pd.DataFrame:
        """
        Export entire collection as dataframe
    
        """
        try:

            if database_name is None:
                collecion = self.mongo_client.database[collection_name]

            else:
                collecion = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collecion.find()))

            logging.info(df.head())
            if '_id' in df.columns.to_list():
                df = df.drop(columns=['_id'], axis=1)

            return df
        
    
        except Exception as e:
            raise SensorException(e,sys)



import pymongo
import os
import certifi
from sensor.constant.database import DATABASE_NAME


ca = certifi.where()

class MongoDBClient:
    client = None
    def __init__(self, database_name=DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                MongoDBClient.client = pymongo.MongoClient(os.getenv('MONGO_DB_URL'), tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise e

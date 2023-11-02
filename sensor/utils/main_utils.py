
import numpy as np
import dill
from sensor.logger import logging


from sensor.exception import SensorException
import yaml
import os, sys

def read_yaml_file(file_path)->dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SensorException(e,sys)
    

def write_yaml_file(file_path, content:object, replace:bool = False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok = True)

        with open(file_path,'w') as file:
            yaml.dump(content,file)
    except Exception as e:
        raise SensorException(e,sys)
    


def save_numpy_array_data(file_path:str, array:np.array)->None:
    """
    save numpy array on the location
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with  open(file_path, 'wb') as file_:
            np.save(file_, array)
    except Exception as e:
        raise SensorException(e,sys)
    

def load_numpy_array_data(file_path:str)->np.array:
    """
    This will load the numpy array data
    """
    try:
        with open(file_path, 'rb') as file_:
            return np.load(file_)
    except Exception as e:
        raise SensorException(e,sys)


def save_object(file_path:str, obj:object)->object:
    try:
        logging.info('ENtered the save_object mthod of mainutiles')
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'wb') as file_:
            dill.dump(obj, file_)
    except Exception as e:
        raise SensorException(e,sys)
    


def load_object(file_path:str)->object:
    """
    This will load the object 
    """
    try:
        logging.info('Entered into load the object')
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} doesn't exist")
        logging.info('path exist')
        with open(file_path, 'rb') as file_object:
            return dill.load(file_object)
    except Exception as e:
        raise SensorException(e,sys)

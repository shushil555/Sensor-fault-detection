from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from sensor.pipeline.training_pipeline import TrainPipeline
from fastapi import FastAPI,File,UploadFile,Request,HTTPException
from sensor.constant.application import APP_PORT,APP_HOST
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.ml.model.estimator import ModelResolver,TargetMapping
from sensor.utils.main_utils import load_object
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get('/train')
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response('Training is already running')
        train_pipeline.run_pipeline()
        return Response('Training successful')
    except  Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get('/predict')
async def predict_route(request:Request,file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exist():
            return Response('Model is not trained yet.')
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(best_model_path)
        y_pred = model.predict(df)
        df['predicted_columns'] = y_pred
        df['predicted_columns'].replace(TargetMapping.reverse_mapping(), inplace = True)
        return df.to_html()        
    except Exception as e:
        return Response(f"Error Occurred! {e}")

if __name__ =='__main__':
    app_run(app, host = APP_HOST, port = APP_PORT)
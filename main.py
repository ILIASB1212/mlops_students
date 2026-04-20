from fastapi import FastAPI
from src.componnent.model_trainner import ModelTrainerConfig, ModelTrainer
import uvicorn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline import predict_pipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app=FastAPI()




@app.post("/")
def index(request: CustomData):
    pipeline = PredictPipeline(request)
    features = pipeline.get_data_as_data_frame()
    prediction = pipeline.predict(features)
    return {"prediction": float(prediction[0])}  
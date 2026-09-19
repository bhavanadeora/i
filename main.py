from fastapi import FastAPI
from pydantic import BaseModel
import joblib as j
 
# Initialize app
app = FastAPI(title="Iris Prediction API")
 
# Load model
model = j.load("svc.pkl")
 
# Define schema
class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
 
# Home route
@app.get("/")
def home():
    return {"message": "Iris FastAPI is running 🌸"}
 
# Prediction route
# @app.post("/predict")
# def predict(data: Iris):   # ✅ FIXED HERE
#     try:
#         features = [[
#             data.sepal_length,
#             data.sepal_width,
#             data.petal_length,
#             data.petal_width
#         ]]
 
#         prediction = model.predict(features)
 
#         return {
#             "prediction": int(prediction[0]),
#             "species": ["setosa", "versicolor", "virginica"][prediction[0]]
#         }
 
#     except Exception as e:
#         return {"error": str(e)}
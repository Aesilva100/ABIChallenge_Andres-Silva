from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import sqlite3

# Define the data model for input data for prediction
class IrisModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize the FastAPI app
app = FastAPI()

# Load the Iris model and target names
try:
    model = joblib.load("model/gaussian_nb_model.pkl")
    # Define the target names
    target_names = ['setosa', 'versicolor', 'virginica']
except FileNotFoundError:
    print("The model file was not found. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# Create a connection to the SQLite database
conn = sqlite3.connect('predictions.db', check_same_thread=False)
c = conn.cursor()
# Create a table to store predictions if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (sepal_length REAL, sepal_width REAL, petal_length REAL, petal_width REAL, prediction TEXT)''')
conn.commit()

# Endpoint for individual predictions
@app.post("/predict/")
async def predict(iris: IrisModel):
    try:
        data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
        prediction_index = model.predict(data)[0]
        prediction_name = target_names[prediction_index]
        # Store the prediction in the database
        c.execute("INSERT INTO predictions VALUES (?,?,?,?,?)", (iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width, prediction_name))
        conn.commit()
        return {"prediction": prediction_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# Endpoint for batch predictions
@app.post("/predict_batch/")
async def predict_batch(iris_list: list[IrisModel]):
    try:
        predictions = []
        for iris in iris_list:
            data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
            prediction_index = model.predict(data)[0]
            prediction_name = target_names[prediction_index]
            predictions.append(prediction_name)
            # Store the prediction in the database
            c.execute("INSERT INTO predictions VALUES (?,?,?,?,?)", (iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width, prediction_name))
            conn.commit()
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during batch prediction: {e}")


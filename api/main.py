from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
#endpoint = "https://localhost:8501/v1/models/potatoes_model:predict"

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


MODEL = tf.keras.models.load_model("C:/Users/Apurva/Desktop/Projects/potato_disease_classification/saved_models/1/v0.1.h5")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "I am pinging..."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)    #after : is for validation, inbuilt in fastapi
):
    #await - while 1st req take 2s to read, it will put this func in suspend mode
    #to serve the 2nd req
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])] #argmax returns the index of the max value
    confidence = np.max(prediction[0]) #returns the max vaue from the array

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
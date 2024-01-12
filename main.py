from fastapi import FastAPI, File, UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras.models import load_model
from PIL import Image
import numpy as np
from uvicorn import run
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

model_dir = "digit_model.h5"
model = load_model(model_dir)


@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )


@app.post("/net/image/prediction/")
async def get_net_image_prediction(image: UploadFile = File(...)):
    with Image.open(image.file) as img:

        height, width, _ = np.array(img).shape
        new_width = 28
        new_height = new_width * height / width

        new_height = 28
        new_width = new_height * width / height
        im = img.resize((int(new_width), int(new_height)), Image.LANCZOS)
        predictions = model.predict(np.transpose(np.array(im), (2, 0, 1)))
        predicted_label = np.argmax(predictions)

    return {
        "model-prediction": predicted_label.tolist(),

    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)

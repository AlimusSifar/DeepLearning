from io import BytesIO
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from PIL import Image
from uvicorn import run

SAVED_MODELS_PATH = "saved_models/potato_disease_classification/"
MODEL_VERSION = "v1.0"
CLASS_NAMES = ["Early blight", "Healthy", "Late blight"]
MODEL = tf.keras.models.load_model(SAVED_MODELS_PATH + MODEL_VERSION)

app = FastAPI()


@app.get("/")
async def ping() -> dict[str, str]:
    return {"message": "Welcome to FastAPI server."}


@app.post("/predict")
async def predict(
    file: UploadFile = File(
        ..., description="Image file to predict. Supported formats: PNG, JPG, JPEG."
    )
) -> dict[str, Any]:
    def read_image_file_as_numpy(file_contents) -> np.ndarray:
        return np.expand_dims(np.array(Image.open(BytesIO(file_contents))), axis=0)

    if file.content_type not in ["image/png", "image/jpg", "image/jpeg"]:
        raise HTTPException(
            status_code=415,
            detail=[
                {
                    "loc": ["body", "file"],
                    "msg": "Unsupported Media Type",
                    "type": "value_error.unsupported_media_type",
                }
            ],
        )

    # get all file details
    file_details = {
        "filename": file.filename,
        "mime_type": file.content_type,
    }
    img_np = read_image_file_as_numpy(await file.read())
    predictions = MODEL.predict(img_np)

    return {
        "message": "Image file received.",
        "file_details": file_details,
        "image_details": {"shape": str(img_np.shape)},
        "prediction": CLASS_NAMES[np.argmax(predictions)],
        "accuracy": f"{np.max(predictions) * 100:.4f}%",
    }


if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=8000, reload=True)

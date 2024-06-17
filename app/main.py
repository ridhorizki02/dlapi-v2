from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from inference import predict_thermal_image
from preprocessing import preprocess_thermal_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post('/predict/multiple')
async def detect_thermal_images(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # Save the uploaded file to a temporary directory
        temp_path = f"tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Preprocess the image
        preprocessed_image = preprocess_thermal_image(temp_path)

        # Perform the detection
        predicted_class, predicted_prob = predict_thermal_image(preprocessed_image)

        formatted_prob = "{:.2f}%".format(predicted_prob)


        results.append({
            "detection_id": len(results) + 1,
            "filename": file.filename,
            "predicted_class": predicted_class,
            "predicted_probability": formatted_prob
        })

    return results
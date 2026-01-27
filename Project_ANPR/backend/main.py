from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from typing import Optional
import cv2, numpy as np, os
from datetime import datetime
import time
from sqlmodel import select
from database import SessionDep, Detections, create_db

create_db()

app = FastAPI(title="AI Vision API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../weights/combined_best.pt")
SAVE_DIR = os.path.join(BASE_DIR, "storage/images")
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def detect_image(image_bytes):
    start = time.time()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    results = model(img)

    annotated = results[0].plot()
    inference_time = (time.time() - start) * 1000
    boxes = results[0].boxes
    num_detections = len(boxes)
    classes = [model.names[int(cls)] for cls in boxes.cls]
    confidence_avg = float(boxes.conf.mean()) if num_detections > 0 else 0.0

    metadata = {
        "inference_time_ms": inference_time,
        "num_detections": num_detections,
        "classes_detected": ",".join(classes),
        "confidence_avg": confidence_avg,
    }
    return annotated, metadata

def save_image(image, file_name):
    filename = f"{file_name}_result.jpg"
    path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, image)
    return filename   # return only filename

@app.post("/detect-image")
async def detect(
    custom_file_name: Optional[str] = Query(None), 
    file: UploadFile = File(...), 
    session: SessionDep = None
):
    image_bytes = await file.read()

    image_name = file.filename
    annotated, meta = detect_image(image_bytes)
    
    if custom_file_name:
        object_name = save_image(annotated, custom_file_name)
    else:
        object_name = save_image(annotated, image_name)

    record = Detections(
        filename=object_name,
        inference_time_ms=meta["inference_time_ms"],
        num_detections=meta["num_detections"],
        classes_detected=meta["classes_detected"],
        confidence_avg=meta["confidence_avg"],
    )

    session.add(record)
    session.commit()

    return JSONResponse({
        "message": "Detections done",
        "download_url": f"/download/{object_name}"
    })

'''
for the below endpoint, fetcg the downloaded image through file id...!
'''
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(SAVE_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="image/jpeg", filename=filename)

@app.get("/detections")
def get_detections(
    session: SessionDep, 
    file_name: Optional[int] = None, 
    file_id: Optional[str] = None
):
    if file_id is not None:
        record = session.get(Detections, file_id)
        if not record:
            raise HTTPException(status_code=404, detail="Detection not found")
        return record

    if file_name is not None:
        query = select(Detections).where(Detections.filename == file_name)
        records = session.exec(
            query
        ).all()
        return records

    records = session.exec(select(Detections)).all()
    return records
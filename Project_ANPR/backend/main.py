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
IMG_SAVE_DIR = os.path.join(BASE_DIR, "storage/images")
VID_SAVE_DIR = os.path.join(BASE_DIR, "storage/videos")

if not IMG_SAVE_DIR:
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)

if not VID_SAVE_DIR:
    os.makedirs(VID_SAVE_DIR, exist_ok=True)

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
    filepath = os.path.join(IMG_SAVE_DIR, filename)
    cv2.imwrite(filepath, image)
    return filename, filepath   # return only filename

IMG_EXT = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".svg"]

@app.post("/detect-image")
async def detect(
    file: UploadFile = File(...), 
    custom_file_name: Optional[str] = Query(None), 
    session: SessionDep = None
    ):
    image_bytes = await file.read()

    name, ext = os.path.splitext(file.filename)
    ext = ext.lower()

    if ext not in IMG_EXT:
        raise HTTPException(status_code=400, detail="Only image files allowed")

    annotated, meta = detect_image(image_bytes)
    image_name = file.filename
    
    if custom_file_name:
        object_name, file_path = save_image(annotated, custom_file_name)
    else:
        object_name, file_path = save_image(annotated, image_name)

    total_detections = meta["num_detections"]

    record = Detections(
        filename=object_name,
        filepath=file_path,
        inference_time_ms=meta["inference_time_ms"],
        num_detections=meta["num_detections"],
        classes_detected=meta["classes_detected"],
        confidence_avg=meta["confidence_avg"],
    )

    session.add(record)
    session.commit()

    return JSONResponse({
        "message": "Detections done",
        "download_url": f"{file_path}",
        "Total Detections": f"{total_detections}",
    })

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
        records = session.exec(query).all()
        return records

    records = session.exec(select(Detections)).all()
    return records

@app.get("/download/{filename}")
def download_file(
    session: SessionDep,
    file_name: Optional[str] = None, 
    file_id: Optional[int] = None, 
    ):
    if file_name is not None:
        query = select(Detections).where(Detections.filename == file_name)
        record = session.exec(query).first()

    elif file_id is not None:
        record = session.get(Detections, file_id)

    else:
        raise HTTPException(400, "Provide file_id or file_name")

    path = record.file_path

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, media_type="image/jpeg", filename=record.filename)

@app.delete("/detections/all")
def delete_all_detections(session: SessionDep):
    records = session.exec(select(Detections)).all()

    for rec in records:
        if os.path.exists(rec.filepath):
            os.remove(rec.filepath)   # delete image file too
        session.delete(rec)

    session.commit()
    return {"message": "All detections cleared"}

@app.delete("/detections/id/{file_id}")
def delete_by_id(
    session: SessionDep,
    file_name: Optional[str] = None, 
    file_id: Optional[int] = None, 
    ):
    if file_name is not None:
        query = select(Detections).where(Detections.filename == file_name)
        record = session.exec(query).first()

    elif file_id is not None:
        record = session.get(Detections, file_id)

    else:
        raise HTTPException(400, "Provide file_id or file_name")

    if not record:
        raise HTTPException(404, "Detection not found")

    if os.path.exists(record.filepath):
        os.remove(record.filepath)

    session.delete(record)
    session.commit()

    return {"message": f"Detection {file_id} deleted"}


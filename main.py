import os
import io
import json
import base64
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
from PIL import Image
from supabase import create_client, Client

app = FastAPI(title="SnapTrace Face API")

# CORS - allow your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase: Optional[Client] = None

def get_sb():
    global supabase
    if supabase is None and SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


# ── MODELS ──
class FaceResult(BaseModel):
    photo_id: str
    photo_url: str
    confidence: float

class MatchResponse(BaseModel):
    faces_detected: int
    matches: List[FaceResult]
    total_photos_scanned: int


# ── HELPERS ──
def image_from_bytes(data: bytes) -> np.ndarray:
    """Convert image bytes to numpy array for face_recognition."""
    img = Image.open(io.BytesIO(data))
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize if too large (speed up processing)
    max_dim = 1200
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return np.array(img)


def extract_embeddings(image_data: bytes) -> List[List[float]]:
    """Extract face embeddings from an image."""
    img_array = image_from_bytes(image_data)
    # Detect face locations
    face_locations = face_recognition.face_locations(img_array, model="hog")
    if not face_locations:
        return []
    # Get embeddings for each face
    encodings = face_recognition.face_encodings(img_array, face_locations)
    return [enc.tolist() for enc in encodings]


def compare_faces(selfie_embedding: List[float], stored_embedding: List[float], tolerance: float = 0.55) -> float:
    """Compare two face embeddings. Returns confidence score 0-100."""
    selfie_np = np.array(selfie_embedding)
    stored_np = np.array(stored_embedding)
    # Euclidean distance
    distance = np.linalg.norm(selfie_np - stored_np)
    # Convert distance to confidence (lower distance = higher confidence)
    # Typical match: distance < 0.6, perfect match: distance ~0.3
    if distance > 0.8:
        return 0.0
    confidence = max(0, (0.8 - distance) / 0.8) * 100
    return round(confidence, 1)


# ── ENDPOINTS ──

@app.get("/")
def health():
    return {"status": "ok", "service": "SnapTrace Face API"}


@app.post("/extract-faces")
async def extract_faces(
    photo_id: str = Form(...),
    event_id: str = Form(...),
    photo_url: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Called when photographer uploads a photo.
    Extracts face embeddings and stores them in Supabase.
    """
    try:
        image_data = await file.read()
        embeddings = extract_embeddings(image_data)
        
        sb = get_sb()
        if sb and embeddings:
            # Store each face embedding
            for i, emb in enumerate(embeddings):
                sb.table("face_embeddings").insert({
                    "photo_id": photo_id,
                    "event_id": event_id,
                    "photo_url": photo_url,
                    "embedding": json.dumps(emb),
                    "face_index": i
                }).execute()
        
        return {
            "photo_id": photo_id,
            "faces_detected": len(embeddings),
            "status": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-face", response_model=MatchResponse)
async def match_face(
    event_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Called when guest uploads a selfie.
    Compares against all stored face embeddings for the event.
    Returns matching photos with confidence scores.
    """
    try:
        image_data = await file.read()
        selfie_embeddings = extract_embeddings(image_data)
        
        if not selfie_embeddings:
            return MatchResponse(
                faces_detected=0,
                matches=[],
                total_photos_scanned=0
            )
        
        # Use the first (largest) face from selfie
        selfie_emb = selfie_embeddings[0]
        
        # Get all face embeddings for this event from Supabase
        sb = get_sb()
        if not sb:
            raise HTTPException(status_code=500, detail="Database not configured")
        
        result = sb.table("face_embeddings").select("*").eq("event_id", event_id).execute()
        stored_faces = result.data if result.data else []
        
        # Compare selfie against each stored face
        matches = {}
        for face_record in stored_faces:
            stored_emb = json.loads(face_record["embedding"])
            confidence = compare_faces(selfie_emb, stored_emb)
            
            if confidence > 40:  # Minimum threshold
                pid = face_record["photo_id"]
                # Keep highest confidence per photo
                if pid not in matches or confidence > matches[pid]["confidence"]:
                    matches[pid] = {
                        "photo_id": pid,
                        "photo_url": face_record["photo_url"],
                        "confidence": confidence
                    }
        
        # Sort by confidence descending
        sorted_matches = sorted(matches.values(), key=lambda x: x["confidence"], reverse=True)
        
        return MatchResponse(
            faces_detected=len(selfie_embeddings),
            matches=[FaceResult(**m) for m in sorted_matches],
            total_photos_scanned=len(set(f["photo_id"] for f in stored_faces))
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-faces-base64")
async def extract_faces_base64(
    photo_id: str = Form(...),
    event_id: str = Form(...),
    photo_url: str = Form(...),
    image_base64: str = Form(...)
):
    """Alternative endpoint accepting base64 image data."""
    try:
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        embeddings = extract_embeddings(image_data)
        
        sb = get_sb()
        if sb and embeddings:
            for i, emb in enumerate(embeddings):
                sb.table("face_embeddings").insert({
                    "photo_id": photo_id,
                    "event_id": event_id,
                    "photo_url": photo_url,
                    "embedding": json.dumps(emb),
                    "face_index": i
                }).execute()
        
        return {"photo_id": photo_id, "faces_detected": len(embeddings), "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-face-base64")
async def match_face_base64(
    event_id: str = Form(...),
    image_base64: str = Form(...)
):
    """Alternative endpoint accepting base64 selfie."""
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        selfie_embeddings = extract_embeddings(image_data)
        
        if not selfie_embeddings:
            return {"faces_detected": 0, "matches": [], "total_photos_scanned": 0}
        
        selfie_emb = selfie_embeddings[0]
        
        sb = get_sb()
        if not sb:
            raise HTTPException(status_code=500, detail="Database not configured")
        
        result = sb.table("face_embeddings").select("*").eq("event_id", event_id).execute()
        stored_faces = result.data or []
        
        matches = {}
        for face_record in stored_faces:
            stored_emb = json.loads(face_record["embedding"])
            confidence = compare_faces(selfie_emb, stored_emb)
            if confidence > 40:
                pid = face_record["photo_id"]
                if pid not in matches or confidence > matches[pid]["confidence"]:
                    matches[pid] = {"photo_id": pid, "photo_url": face_record["photo_url"], "confidence": confidence}
        
        sorted_matches = sorted(matches.values(), key=lambda x: x["confidence"], reverse=True)
        
        return {
            "faces_detected": len(selfie_embeddings),
            "matches": sorted_matches,
            "total_photos_scanned": len(set(f["photo_id"] for f in stored_faces))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

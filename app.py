import os
import sys
import shutil
import subprocess
import uuid
import csv
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
Path("static").mkdir(exist_ok=True)
Path("runs").mkdir(exist_ok=True)
Path("tmp").mkdir(exist_ok=True)


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    # Save the uploaded video temporarily
    ext = os.path.splitext(video.filename)[1] or ".mp4"
    unique_id = uuid.uuid4().hex
    tmp_filename = f"val_vid_{unique_id}{ext}"
    tmp_path = Path("tmp") / tmp_filename
    
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    try:
        # Run the hybrid pipeline
        cmd = [
            sys.executable, "hybrid_pipeline.py",
            "--video", str(tmp_path),
            "--output-dir", "runs/webapp_runs"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            return JSONResponse(
                status_code=500, 
                content={"error": "Pipeline failed", "details": process.stderr}
            )
            
        # Parse the output to find the CSV path and run directory
        output_txt = process.stdout
        csv_path = None
        for line in output_txt.splitlines():
            if "Instance log    :" in line:
                csv_path = line.split(":", 1)[1].strip()
                break
                
        if not csv_path or not os.path.exists(csv_path):
            return JSONResponse(
                status_code=500,
                content={"error": "Could not find pipeline output log.", "details": output_txt}
            )
            
        # Read the CSV to get instances
        instances = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # crop_path is absolute or relative, but to serve it statically we map it relative to root
                # Usually it looks like: runs/webapp_runs/.../crops/instance_0001.jpg
                # We can just normalize it.
                crop_path = Path(row["crop_path"])
                # Extract path relative to workspace
                # Depending on how pipeline outputted it, if it's absolute, resolve it relative to cwd
                try:
                    rel_crop = crop_path.relative_to(Path.cwd())
                except ValueError:
                    # If it's already relative or somewhere else, try to find "runs/" in it
                    # fallback to string parsing
                    parts = crop_path.parts
                    if "runs" in parts:
                        idx = parts.index("runs")
                        rel_crop = Path(*parts[idx:])
                    else:
                        rel_crop = crop_path
                        
                instances.append({
                    "id": row.get("instance_id", ""),
                    "confidence": float(row.get("best_confidence", 0.0)),
                    "first_seconds": row.get("first_seconds", ""),
                    "crop_url": "/" + str(rel_crop).replace(os.sep, "/")
                })
                
        return {
            "success": True,
            "total_potholes": len(instances),
            "instances": instances
        }
    finally:
        # Cleanup temp upload file if exists
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Static file mounts must come AFTER all API routes so that the "/" mount
# does not intercept API requests before they reach their handlers.
app.mount("/runs", StaticFiles(directory="runs"), name="runs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

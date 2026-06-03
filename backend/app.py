from __future__ import annotations  # Allows modern type hints

import time          # Used to measure processing time
import uuid          # Used to create unique IDs for files
import sys
from pathlib import Path  # Helps manage file paths easily

import cv2           # OpenCV used to save images
from flask import Flask, render_template, request, send_from_directory  # Web framework
from werkzeug.utils import secure_filename  # Makes filenames safe


# Ensure parent directory is in Python path so backend imports work correctly
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ➤ This fixes import issues so backend files can be accessed


# Import project-specific model service and paths
from backend.model.demo2 import ROOT_DIR, process_image
# ➤ ROOT_DIR = main project folder
# ➤ process_image = function that runs AI model


# Define important project directories
APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"      # HTML files
STATIC_DIR = ROOT_DIR / "static"          # CSS, JS, results
UPLOADS_DIR = ROOT_DIR / "uploads"        # Uploaded images
RESULTS_DIR = STATIC_DIR / "results"      # Output images
# ➤ These paths organize files properly


def create_app() -> Flask:
    """
    Factory function to create and configure Flask app.
    """

    # Create Flask app with custom folders
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
        template_folder=str(FRONTEND_DIR),
    )
    # ➤ Sets where static files and HTML templates are stored

    # Limit max upload size (25 MB)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    # ➤ Prevents very large file uploads

    # Ensure required directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # ➤ Creates folders if they don’t exist

    # --------------------------
    # ROUTE: Home Page
    # --------------------------
    @app.get("/")
    def index() -> str:
        return render_template("index.html")
    # ➤ Loads main webpage

    # --------------------------
    # ROUTE: Serve frontend files
    # --------------------------
    @app.get("/frontend/<path:filename>")
    def frontend_assets(filename: str):
        return send_from_directory(FRONTEND_DIR, filename)
    # ➤ Sends CSS/JS files to browser

    # --------------------------
    # ROUTE: Health check API
    # --------------------------
    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200
    # ➤ Simple API to check if server is running

    # --------------------------
    # ROUTE: Main AI Detection API
    # --------------------------
    @app.post("/detect")
    def detect() -> tuple[dict[str, object], int]:

        # Check if image is sent
        if "image" not in request.files:
            return {"error": "No image file provided."}, 400

        file = request.files["image"]

        # Check if filename exists
        if not file.filename:
            return {"error": "Empty filename."}, 400

        # Get parameters from frontend
        min_area = int(request.form.get("min_area", 300))
        max_area = int(request.form.get("max_area", 200000))
        model_type = request.form.get("model_type", "vit_b")
        checkpoint_path = request.form.get("checkpoint_path") or None
        # ➤ These control filtering and model behavior

        # Validate values
        if min_area < 1 or max_area <= min_area:
            return {"error": "Invalid area thresholds."}, 400

        # Create unique ID for request
        request_id = uuid.uuid4().hex

        # Make filename safe
        upload_name = secure_filename(file.filename)

        # Save uploaded file
        upload_path = UPLOADS_DIR / f"{request_id}_{upload_name}"
        file.save(upload_path)
        # ➤ Stores uploaded image

        # Start timer
        started = time.perf_counter()

        try:
            # Call AI model
            result = process_image(
                upload_path.read_bytes(),
                min_area=min_area,
                max_area=max_area,
                model_type=model_type,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            return {"error": str(exc)}, 500
        # ➤ Runs YOLO model 

        # Calculate processing time
        processing_ms = round((time.perf_counter() - started) * 1000, 2)

        # Output filenames
        annotated_name = f"{request_id}_annotated.png"
        mask_name = f"{request_id}_mask.png"

        # Save result paths
        annotated_path = RESULTS_DIR / annotated_name
        mask_path = RESULTS_DIR / mask_name

        # Save images using OpenCV
        cv2.imwrite(str(annotated_path), result["annotated_bgr"])
        cv2.imwrite(str(mask_path), result["mask_binary"])
        # ➤ Saves processed images

        # Return response to frontend
        return (
            {
                "building_count": result["count"],  # total buildings
                "filtered_mask_count": result["filtered_mask_count"],
                "processing_time_ms": processing_ms,
                "annotated_image_url": f"/static/results/{annotated_name}",
                "mask_image_url": f"/static/results/{mask_name}",
                "original_filename": upload_name,
            },
            200,
        )
        # ➤ Sends results back to frontend

    # --------------------------
    # ROUTE: Download results
    # --------------------------
    @app.get("/downloads/<path:filename>")
    def downloads(filename: str):
        return send_from_directory(RESULTS_DIR, filename, as_attachment=True)
    # ➤ Allows user to download result images

    return app


# Create Flask app
app = create_app()
# ➤ Initializes app


# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
# ➤ Starts server at localhost:5000
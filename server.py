"""
FastAPI Server for Auto Pipeline UI
Wraps auto_pipeline.py with a web interface
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import subprocess
import shutil
import zipfile
import os
import uuid
from pathlib import Path
import tempfile

app = FastAPI(title="Auto Dataset Generator")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/api/generate")
async def generate_dataset(
    zip_file: UploadFile = File(...),
    output_folder: str = Form(...),
    num_classes: int = Form(...),
    class_names: str = Form(...)
):
    """
    Main endpoint: Accept ZIP, run pipeline, return result ZIP
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Save uploaded ZIP
        zip_path = session_dir / "input.zip"
        with open(zip_path, "wb") as f:
            content = await zip_file.read()
            f.write(content)
        
        # 2. Extract ZIP
        input_images_dir = session_dir / "images"
        input_images_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_images_dir)
        
        # 3. Prepare output directory (absolute path)
        output_dir = session_dir / output_folder
        
        # 4. Run auto_pipeline.py
        class_list = class_names.strip().split()
        
        if len(class_list) != num_classes:
            raise HTTPException(
                status_code=400, 
                detail=f"Number of class names ({len(class_list)}) doesn't match num_classes ({num_classes})"
            )
        
        # Build command
        cmd = [
            "python",
            "auto_pipeline.py",
            str(input_images_dir),
            str(output_dir),
        ] + class_list
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Expected output directory: {output_dir}")
        
        # Execute pipeline with UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {result.stderr}"
            )
        
        print(f"Pipeline stdout: {result.stdout}")
        print(f"Pipeline stderr: {result.stderr}")
        
        # Check if output was created in project root instead
        project_output = Path(os.getcwd()) / output_folder
        if project_output.exists() and not output_dir.exists():
            print(f"⚠ Pipeline created output in project root, moving to session dir")
            shutil.move(str(project_output), str(output_dir))
        
        # 5. Verify output directory was created AND contains data
        if not output_dir.exists():
            raise HTTPException(
                status_code=500,
                detail="Output directory was not created by pipeline"
            )
        
        # Check that output has actual content (subdirectories for classes)
        subdirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'splits']
        if len(subdirs) == 0:
            raise HTTPException(
                status_code=500,
                detail=f"Output directory exists but contains no class folders. Check if pipeline completed successfully."
            )
        
        print(f"✓ Output verified: {len(subdirs)} class folders found")
        
        # List what's in the output directory for debugging
        all_items = list(output_dir.iterdir())
        print(f"Output directory contents: {[item.name for item in all_items]}")
        
        # Count images in each class folder
        for subdir in subdirs:
            image_count = len(list(subdir.glob('*.*')))
            print(f"  {subdir.name}: {image_count} files")
        
        # 6. Zip the output - zip from parent directory to preserve folder structure
        output_zip = session_dir / f"{output_folder}.zip"
        print(f"Creating ZIP at: {output_zip}")
        print(f"Zipping from root_dir={output_dir.parent}, base_dir={output_dir.name}")
        
        shutil.make_archive(
            str(output_zip.with_suffix('')),
            'zip',
            root_dir=output_dir.parent,
            base_dir=output_dir.name
        )
        
        print(f"✓ ZIP created successfully: {output_zip.name}")
        
        # 7. Return the ZIP file with proper headers to force download
        return FileResponse(
            path=output_zip,
            filename=f"{output_folder}.zip",
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{output_folder}.zip"'
            }
        )
        
    except Exception as e:
        # Cleanup on error
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Auto Dataset Generator is running"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Auto Dataset Generator Server")
    print("=" * 60)
    print("📁 Starting server...")
    print("🌐 Open: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

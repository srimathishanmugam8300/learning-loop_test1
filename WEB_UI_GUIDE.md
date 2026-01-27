# 🌐 Web UI Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn python-multipart
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python server.py
```

You should see:
```
🚀 Auto Dataset Generator Server
📁 Starting server...
🌐 Open: http://localhost:8000
```

### 3. Open Your Browser

Navigate to: **http://localhost:8000**

---

## How to Use

### Step-by-Step:

1. **Prepare Your Images**
   - Collect all unlabeled images in a folder
   - Zip the folder (right-click → Send to → Compressed folder)

2. **Upload the ZIP**
   - Click the upload area or drag & drop your ZIP file

3. **Enter Dataset Info**
   - **Number of Classes**: How many categories (e.g., 7)
   - **Class Names**: Space-separated names (e.g., `cat dog bird fish`)
   - **Output Folder Name**: Name for your dataset (e.g., `animals_dataset`)

4. **Generate**
   - Click "🚀 Generate Dataset"
   - Wait for processing (may take a few minutes)

5. **Download**
   - Click "📥 Download Result" when ready
   - Extract the ZIP to see your organized dataset

---

## What You Get

The downloaded ZIP contains:
```
output_folder/
├── metadata.json          # Dataset info
├── report.txt            # Summary statistics
├── class_1/              # Organized images
├── class_2/
└── splits/
    ├── train/
    ├── val/
    └── test/
```

---

## Architecture

```
User Browser
    ↓ (uploads ZIP)
FastAPI Server (server.py)
    ↓ (extracts & runs)
auto_pipeline.py
    ↓ (generates dataset)
Server zips output
    ↓ (sends back)
User downloads result
```

---

## Files Created

```
project/
├── server.py                 # FastAPI backend
├── static/
│   ├── index.html           # Web interface
│   ├── style.css            # Styling
│   └── script.js            # Client logic
└── uploads/                 # Temporary files (auto-cleaned)
```

---

## Troubleshooting

### Port Already in Use
If port 8000 is busy, edit `server.py` line 105:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change to 8001
```

### "Module not found"
Install missing dependencies:
```bash
pip install fastapi uvicorn python-multipart
```

### Pipeline Fails
Check that:
- ZIP contains valid images
- Number of classes matches class names count
- `auto_pipeline.py` works standalone

---

## Advanced: Running on Network

To access from other devices on your network:

1. Find your local IP:
   ```bash
   ipconfig  # Windows
   ```

2. Start server (it already binds to 0.0.0.0)

3. Access from other devices:
   ```
   http://YOUR_LOCAL_IP:8000
   ```

---

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Processing**: Your existing `auto_pipeline.py` (unchanged)

---

## Notes

✅ Zero changes to `auto_pipeline.py`
✅ Works 100% locally
✅ No cloud dependencies
✅ Clean separation of concerns
✅ Production-ready architecture

---

**Happy Dataset Generation! 🚀**

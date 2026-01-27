# 🎨 Web UI - Auto Dataset Generator

A clean web interface for your ML dataset generation pipeline.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install fastapi uvicorn python-multipart

# 2. Start server
python server.py

# 3. Open browser
# Navigate to: http://localhost:8000
```

## ✨ Features

- ✅ **Drag & Drop** ZIP upload
- ✅ **Real-time** validation
- ✅ **Progress** indicators
- ✅ **One-click** download
- ✅ **Clean** modern UI
- ✅ **Zero modifications** to core pipeline

## 📁 Project Structure

```
learning_loop_T1/
│
├── server.py              # FastAPI backend (NEW)
├── auto_pipeline.py       # Your existing ML pipeline (UNCHANGED)
│
├── static/                # Frontend (NEW)
│   ├── index.html        # Web interface
│   ├── style.css         # Styling
│   └── script.js         # Client logic
│
└── uploads/               # Temporary storage (auto-created)
```

## 🎯 How It Works

```
┌─────────────┐
│   Browser   │ ──► Upload ZIP + Config
└─────────────┘
       │
       ▼
┌─────────────┐
│  FastAPI    │ ──► Extract & Validate
│  Server     │
└─────────────┘
       │
       ▼
┌─────────────┐
│auto_pipeline│ ──► Generate Dataset
│    .py      │
└─────────────┘
       │
       ▼
┌─────────────┐
│  Zip Result │ ──► Return to Browser
└─────────────┘
```

## 🎮 Usage Example

1. **Prepare**: Zip your unlabeled images folder
2. **Upload**: Drag ZIP to web interface
3. **Configure**:
   - Number of classes: `5`
   - Class names: `cat dog bird fish lion`
   - Output folder: `animals_dataset`
4. **Generate**: Click button & wait
5. **Download**: Get organized dataset as ZIP

## 🛠️ Technical Details

### Backend (server.py)
- **Framework**: FastAPI
- **Features**:
  - File upload handling
  - ZIP extraction
  - Subprocess management
  - Result packaging
  - Session management (UUID-based)

### Frontend
- **Pure**: HTML/CSS/JavaScript (no frameworks)
- **Features**:
  - Drag & drop
  - Form validation
  - Loading states
  - Error handling
  - Responsive design

### Integration
- Wraps `auto_pipeline.py` via subprocess
- No code changes to existing pipeline
- Clean separation of concerns

## 📦 Dependencies

```txt
fastapi>=0.104.0        # Web framework
uvicorn>=0.24.0         # ASGI server
python-multipart>=0.0.6 # File upload support
```

## 🔧 Configuration

Edit `server.py` to customize:

```python
# Change port (line 105)
uvicorn.run(app, host="0.0.0.0", port=8000)

# Change upload directory (line 18)
UPLOADS_DIR = Path("uploads")
```

## 🌐 Network Access

Access from other devices on your network:

1. Find your IP: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
2. Server already binds to `0.0.0.0`
3. Access from other device: `http://YOUR_IP:8000`

## 🐛 Troubleshooting

### Port in use
```bash
# Change port in server.py or kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Module not found
```bash
pip install -r requirements.txt
```

### Pipeline fails
- Verify ZIP contains images
- Check class count matches names
- Test `auto_pipeline.py` standalone first

## 📊 API Endpoints

### `GET /`
Serves the web interface

### `POST /api/generate`
Main endpoint for dataset generation

**Request** (multipart/form-data):
- `zip_file`: ZIP file with images
- `output_folder`: Output folder name
- `num_classes`: Number of classes (integer)
- `class_names`: Space-separated class names

**Response**: ZIP file with organized dataset

### `GET /api/health`
Health check endpoint

**Response**: `{"status": "ok", "message": "..."}`

## 🎨 UI Design

- **Colors**: Purple gradient theme
- **Typography**: System fonts (SF Pro, Segoe UI, Roboto)
- **Animations**: Smooth transitions & loading states
- **Responsive**: Works on desktop & mobile

## 🔒 Security Notes

- This is designed for **local use**
- No authentication (not needed for localhost)
- For production deployment, add:
  - Rate limiting
  - File size limits
  - Input sanitization
  - HTTPS
  - Authentication

## 📝 License

Same as parent project

## 🙏 Credits

Built on top of the excellent `auto_pipeline.py` ML pipeline.

---

**Enjoy your new web interface! 🎉**

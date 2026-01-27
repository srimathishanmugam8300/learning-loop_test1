# 🚀 QUICK START CARD

## Install (One-Time Setup)

```bash
pip install fastapi uvicorn python-multipart
```

---

## Start Server

**Option A - Easy Way:**
```bash
start_web_ui.bat
```

**Option B - Manual:**
```bash
python server.py
```

---

## Open Browser

```
http://localhost:8000
```

---

## Use the Interface

1. **Upload** ZIP file with images
2. **Enter** number of classes (e.g., `7`)
3. **Type** class names separated by spaces (e.g., `cat dog bird`)
4. **Name** your output folder (e.g., `my_dataset`)
5. **Click** "Generate Dataset"
6. **Wait** for processing (1-5 minutes)
7. **Download** the result ZIP

---

## Stop Server

Press `Ctrl + C` in the terminal

---

## Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Port already in use"
Edit `server.py` line 105, change port to `8001`

### "Pipeline failed"
- Check ZIP contains valid images
- Verify class count matches number of names
- Test standalone: `python auto_pipeline.py ...`

---

## Files Structure

```
📁 learning_loop_T1/
├── 🚀 server.py              # Backend
├── 🔧 auto_pipeline.py       # ML pipeline (unchanged)
├── 📝 start_web_ui.bat       # Easy launcher
│
└── 📁 static/                # Frontend
    ├── index.html
    ├── style.css
    └── script.js
```

---

## What You Get

```
output_folder.zip
    └── output_folder/
        ├── metadata.json
        ├── report.txt
        ├── class_1/
        ├── class_2/
        └── splits/
            ├── train/
            ├── val/
            └── test/
```

---

## Example Usage

```
Input ZIP: animals_photos.zip (500 unlabeled images)

Settings:
- Classes: 5
- Names: cat dog bird fish lion
- Output: animals_dataset

Result: animals_dataset.zip with organized files
```

---

## Support

📖 Read [WEB_UI_GUIDE.md](WEB_UI_GUIDE.md) for detailed instructions
📖 Read [WEB_UI_README.md](WEB_UI_README.md) for technical details
📖 Read [WEB_UI_IMPLEMENTATION_SUMMARY.md](WEB_UI_IMPLEMENTATION_SUMMARY.md) for overview

---

**🎉 That's it! Enjoy your new web interface!**

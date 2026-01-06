import os

# ====== CONFIG ======
FOLDER_PATH = r"D:\my_projects\learning_loop_T1\data\images3"   # <- change this

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def rename_images_as_numbers(folder):
    if not os.path.isdir(folder):
        print("Folder not found:", folder)
        return

    files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

    # Sort files to rename in consistent order
    files.sort()

    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1].lower()

        new_name = f"{i}{ext}"
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

    print(f"Renamed {len(files)} images successfully.")

if __name__ == "__main__":
    rename_images_as_numbers(FOLDER_PATH)

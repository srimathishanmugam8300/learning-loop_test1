import os
import uuid

FOLDER_PATH = r"D:\my_projects\learning_loop_T1\data\images4"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def rename_images_to_numbers(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    files.sort()

    print(f"Found {len(files)} images. Renaming safely...\n")

    temp_map = {}

    # STEP 1 — rename all files to TEMP names
    for f in files:
        old_path = os.path.join(folder_path, f)
        ext = os.path.splitext(f)[1].lower()
        temp_name = f"temp_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(folder_path, temp_name)

        os.rename(old_path, temp_path)
        temp_map[temp_name] = ext

    # STEP 2 — rename temp files to 1.jpg, 2.jpg...
    for i, temp_name in enumerate(sorted(temp_map.keys()), start=1):
            ext = temp_map[temp_name]
            final_name = f"{i}{ext}"
            temp_path = os.path.join(folder_path, temp_name)
            final_path = os.path.join(folder_path, final_name)

            os.rename(temp_path, final_path)
            print(f"{final_name}")

    print("\nDone — Renamed all files successfully.")


if __name__ == "__main__":
    rename_images_to_numbers(FOLDER_PATH)

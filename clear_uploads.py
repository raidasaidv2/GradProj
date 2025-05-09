import os

uploads_folder = 'uploads'

for filename in os.listdir(uploads_folder):
    file_path = os.path.join(uploads_folder, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)  # Delete the file
            print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

import os
from werkzeug.utils import secure_filename
from backend.config import Config

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_and_save_upload(file, user_id):
    """
    Validates the uploaded file and saves it securely to a user-specific directory.
    Returns the file path if successful, otherwise raises ValueError.
    """
    if not file or file.filename == '':
        raise ValueError("No file provided.")
    if not allowed_file(file.filename):
        raise ValueError("File type not allowed. Must be PNG, JPG, or JPEG.")

    filename = secure_filename(file.filename)
    user_dir = os.path.join(Config.UPLOAD_FOLDER, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    
    file_path = os.path.join(user_dir, filename)
    file.save(file_path)
    
    # Check max size post-save or during upload
    if os.path.getsize(file_path) > Config.MAX_CONTENT_LENGTH:
        os.remove(file_path)
        raise ValueError("File exceeds maximum allowed size.")

    return file_path

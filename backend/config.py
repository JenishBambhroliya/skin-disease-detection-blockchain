import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'dev-secret-key'
    
    _db_uri = os.environ.get('DATABASE_URI') or 'sqlite:///database/skin_detection.db'
    if _db_uri.startswith('sqlite:///'):
        _path = _db_uri.split('sqlite:///')[-1]
        _db_uri = f"sqlite:///{os.path.join(BASE_DIR, _path)}"
        
    SQLALCHEMY_DATABASE_URI = _db_uri
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload limits and allowed extensions
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    
    # Model
    MODEL_PATH = os.environ.get('MODEL_PATH', 'trained_models/skin_cancer_cnn_model.h5')
    CLASS_LABELS_PATH = os.environ.get('CLASS_LABELS_PATH', 'model/class_labels.json')

    # Security & Blockchain
    AES_SECRET_KEY = os.environ.get('AES_SECRET_KEY', 'default-aes-256-key-must-be-32-by').encode()
    
    # Blockchain
    BLOCKCHAIN_RPC_URL = os.environ.get('BLOCKCHAIN_RPC_URL', 'http://127.0.0.1:7545')
    WALLET_ADDRESS = os.environ.get('WALLET_ADDRESS')
    PRIVATE_KEY = os.environ.get('PRIVATE_KEY')
    CONTRACT_ADDRESS = os.environ.get('CONTRACT_ADDRESS')
    
    # IPFS
    IPFS_HOST = os.environ.get('IPFS_HOST', '/ip4/127.0.0.1/tcp/5001/http')

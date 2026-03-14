import logging
import os

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure the main application logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'app.log'))
    ]
)

app_logger = logging.getLogger('app')

# Create a specialized logger for blockchain errors
blockchain_logger = logging.getLogger('blockchain')
blockchain_handler = logging.FileHandler(os.path.join(log_dir, 'blockchain_failures.log'))
blockchain_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
blockchain_logger.addHandler(blockchain_handler)
blockchain_logger.propagate = False  # Avoid duplicate logging if handled separately

# Create a specialized logger for IPFS errors
ipfs_logger = logging.getLogger('ipfs')
ipfs_handler = logging.FileHandler(os.path.join(log_dir, 'ipfs_failures.log'))
ipfs_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
ipfs_logger.addHandler(ipfs_handler)
ipfs_logger.propagate = False

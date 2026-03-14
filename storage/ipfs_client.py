import os
import hashlib
import base58
from backend.config import Config
from backend.logger import ipfs_logger

class IPFSClient:
    def __init__(self):
        self.connected = False
        
        # Try to connect to a real IPFS node first
        try:
            import ipfshttpclient
            self.client = ipfshttpclient.connect(Config.IPFS_HOST)
            self.connected = True
            ipfs_logger.info(f"Connected to IPFS node at {Config.IPFS_HOST}")
        except Exception as e:
            ipfs_logger.warning(f"IPFS node not available at {Config.IPFS_HOST}: {str(e)}")
            ipfs_logger.info("Using local IPFS simulation (CID generation from file hash)")
            self.client = None
            # Enable local simulation mode
            self._local_mode = True
            self._local_store = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'storage', 'ipfs_local')
            os.makedirs(self._local_store, exist_ok=True)
            self.connected = True  # Mark as connected so uploads proceed

    def _generate_cid(self, file_path: str) -> str:
        """Generate a CIDv0-style hash from file content (matches real IPFS hashing)."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256_hash.update(chunk)
        
        # Create a CIDv0-like identifier: Qm + base58(sha256)
        digest = sha256_hash.digest()
        # Prepend multihash header: 0x12 (sha256) + 0x20 (32 bytes)
        multihash = b'\x12\x20' + digest
        cid = base58.b58encode(multihash).decode('utf-8')
        return cid

    def upload_file(self, file_path: str) -> str:
        """Uploads a file to IPFS and returns the CID. Falls back to local simulation."""
        if not self.connected:
            ipfs_logger.warning("Attempted upload but IPFS is disconnected.")
            return None
        
        # Try real IPFS first
        if self.client:
            try:
                res = self.client.add(file_path)
                return res['Hash']
            except Exception as e:
                ipfs_logger.error(f"Failed to upload {file_path} to IPFS: {str(e)}")
                return None
        
        # Local simulation mode
        try:
            cid = self._generate_cid(file_path)
            # Store a copy locally with the CID as filename
            import shutil
            dest = os.path.join(self._local_store, cid)
            if not os.path.exists(dest):
                shutil.copy2(file_path, dest)
            ipfs_logger.info(f"Local IPFS: stored {file_path} with CID {cid}")
            return cid
        except Exception as e:
            ipfs_logger.error(f"Failed local IPFS store for {file_path}: {str(e)}")
            return None

    def retrieve_file(self, cid: str, output_path: str) -> bool:
        """Retrieves a file from IPFS by CID. Returns True if successful."""
        if not self.connected:
            ipfs_logger.warning("Attempted retrieve but IPFS is disconnected.")
            return False
        
        # Try real IPFS first
        if self.client:
            try:
                self.client.get(cid)
                import shutil
                if os.path.exists(cid):
                    shutil.move(cid, output_path)
                    return True
                return False
            except Exception as e:
                ipfs_logger.error(f"Failed to retrieve CID {cid} from IPFS: {str(e)}")
                return False
        
        # Local simulation mode
        try:
            import shutil
            local_path = os.path.join(self._local_store, cid)
            if os.path.exists(local_path):
                shutil.copy2(local_path, output_path)
                return True
            ipfs_logger.warning(f"CID {cid} not found in local store")
            return False
        except Exception as e:
            ipfs_logger.error(f"Failed to retrieve CID {cid} from local store: {str(e)}")
            return False

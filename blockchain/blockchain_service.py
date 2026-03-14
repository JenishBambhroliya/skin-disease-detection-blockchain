import json
import os
from web3 import Web3
from backend.config import Config
from backend.logger import blockchain_logger

class BlockchainService:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(Config.BLOCKCHAIN_RPC_URL))
        self.connected = self.w3.is_connected()
        self.contract = None

        if not self.connected:
            blockchain_logger.error(f"Failed to connect to blockchain at {Config.BLOCKCHAIN_RPC_URL}")
            return
            
        abi_path = os.path.join(os.path.dirname(__file__), 'contract_abi.json')
        if not os.path.exists(abi_path):
            blockchain_logger.warning("contract_abi.json not found. Deploy contract first.")
            self.connected = False
            return
            
        if not Config.CONTRACT_ADDRESS:
            blockchain_logger.warning("CONTRACT_ADDRESS not configured in .env.")
            self.connected = False
            return
            
        with open(abi_path, 'r') as f:
            abi = json.load(f)
            
        checksum_address = self.w3.to_checksum_address(Config.CONTRACT_ADDRESS.strip().lower())
        self.contract = self.w3.eth.contract(address=checksum_address, abi=abi)
        
        # Use explicit private key if set, else fallback to ganache account
        self.private_key = Config.PRIVATE_KEY.strip() if Config.PRIVATE_KEY else None
        self.wallet_address = Config.WALLET_ADDRESS.strip() if Config.WALLET_ADDRESS else None
        
        # When private key is available, derive wallet address from it
        # to ensure checksum address exactly matches (required by web3 v7+)
        if self.private_key:
            from eth_account import Account
            acct = Account.from_key(self.private_key)
            self.wallet_address = acct.address
        elif not self.wallet_address and self.w3.eth.accounts:
            self.wallet_address = self.w3.eth.accounts[0]

        if self.wallet_address:
            self.wallet_address = self.w3.to_checksum_address(self.wallet_address)

    def add_record(self, user_id: int, image_id: int, sha256_hash: str, ipfs_cid: str) -> str:
        """Adds a record to blockchain. Returns transaction hash or None if fallback."""
        if not self.connected:
            blockchain_logger.warning("Attempted to add record but blockchain is offline/unconfigured. Fallback triggered.")
            return None

        ipfs_cid = ipfs_cid or "no-ipfs-cid"
        
        try:
            tx = self.contract.functions.addRecord(
                user_id, image_id, sha256_hash, ipfs_cid
            ).build_transaction({
                'from': self.wallet_address,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'gas': 2000000,
                'gasPrice': self.w3.to_wei('20', 'gwei')
            })
            
            if self.private_key:
                signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            else:
                # Local testnet without explicit private key requirement
                tx_hash = self.w3.eth.send_transaction({
                    'to': Config.CONTRACT_ADDRESS,
                    'from': self.wallet_address,
                    'data': tx['data']
                })
                
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return self.w3.to_hex(tx_hash)

        except Exception as e:
            blockchain_logger.error(f"Failed to add record for image_id {image_id}: {str(e)}")
            return None

    def get_record(self, image_id: int) -> dict:
        """Retrieves a record from the blockchain. Returns dict or None."""
        if not self.connected:
            blockchain_logger.warning("Attempted to get record but blockchain is offline/unconfigured.")
            return None

        try:
            record = self.contract.functions.getRecord(image_id).call()
            return {
                'userId': record[0],
                'imageId': record[1],
                'sha256Hash': record[2],
                'ipfsCid': record[3],
                'timestamp': record[4],
                'uploader': record[5]
            }
        except Exception as e:
            blockchain_logger.error(f"Failed to get record for image_id {image_id}: {str(e)}")
            return None

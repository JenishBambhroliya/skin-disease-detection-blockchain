import json
import solcx
from web3 import Web3
from backend.config import Config
from backend.logger import blockchain_logger

def deploy():
    try:
        solcx.install_solc('0.8.0')
        with open('blockchain/contract.sol', 'r') as file:
            contract_source_code = file.read()

        compiled_sol = solcx.compile_source(
            contract_source_code,
            output_values=['abi', 'bin'],
            solc_version='0.8.0'
        )

        contract_id, contract_interface = compiled_sol.popitem()
        bytecode = contract_interface['bin']
        abi = contract_interface['abi']

        # Save ABI to a file so that the web app can use it easily
        with open('blockchain/contract_abi.json', 'w') as f:
            json.dump(abi, f)

        w3 = Web3(Web3.HTTPProvider(Config.BLOCKCHAIN_RPC_URL))
        
        if not w3.is_connected():
            print("Failed to connect to blockchain at", Config.BLOCKCHAIN_RPC_URL)
            return

        w3.eth.default_account = w3.eth.accounts[0] if w3.eth.accounts else Config.WALLET_ADDRESS

        print("Deploying contract from account:", w3.eth.default_account)

        SkinDiseaseRecords = w3.eth.contract(abi=abi, bytecode=bytecode)
        tx_hash = SkinDiseaseRecords.constructor().transact()
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        print(f"Contract deployed successfully at address: {tx_receipt.contractAddress}")
        print("Please update CONTRACT_ADDRESS in your .env file.")
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        blockchain_logger.error(f"Deployment failed: {e}")

if __name__ == '__main__':
    deploy()

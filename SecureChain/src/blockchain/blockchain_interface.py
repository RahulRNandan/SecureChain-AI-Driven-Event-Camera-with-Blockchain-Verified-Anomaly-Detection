from web3 import Web3
import json

class BlockchainInterface:
    def __init__(self, contract_address, contract_abi):
        self.web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))  # Example local node
        self.contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)
    
    def log_event(self, event_data, private_key, account_address):
        tx = self.contract.functions.logEvent(event_data).buildTransaction({
            'chainId': 1,  # Mainnet or appropriate network ID
            'gas': 2000000,
            'gasPrice': self.web3.toWei('20', 'gwei'),
            'nonce': self.web3.eth.getTransactionCount(account_address),
        })
        
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return tx_hash.hex()

    def get_event(self, event_id):
        return self.contract.functions.getEvent(event_id).call()

# Usage example
if __name__ == '__main__':
    contract_address = 'your_contract_address'
    contract_abi = json.load(open('path_to_abi.json'))
    
    blockchain = BlockchainInterface(contract_address, contract_abi)
    # Log an event
    private_key = 'your_private_key'
    account_address = 'your_account_address'
    event_data = {"timestamp": 1622540400, "eventType": "motion_detected"}
    tx_hash = blockchain.log_event(event_data, private_key, account_address)
    print(f"Event logged with tx hash: {tx_hash}")

const Web3 = require('web3');
const fs = require('fs');
const path = require('path');
const AccessControlABI = JSON.parse(fs.readFileSync(path.resolve(__dirname, './contracts/AccessControl.json'))).abi;
const EventLoggerABI = JSON.parse(fs.readFileSync(path.resolve(__dirname, './contracts/EventLogger.json'))).abi;

const web3 = new Web3('http://localhost:8545');  // Example for local node

async function deploy() {
    const accounts = await web3.eth.getAccounts();
    const admin = accounts[0];

    // Deploy AccessControl contract
    const accessControl = await new web3.eth.Contract(AccessControlABI)
        .deploy({ data: '0x' + fs.readFileSync(path.resolve(__dirname, './contracts/AccessControl.bin')).toString() })
        .send({ from: admin, gas: 1500000 });
    console.log('AccessControl contract deployed at:', accessControl.options.address);

    // Deploy EventLogger contract
    const eventLogger = await new web3.eth.Contract(EventLoggerABI)
        .deploy({ data: '0x' + fs.readFileSync(path.resolve(__dirname, './contracts/EventLogger.bin')).toString() })
        .send({ from: admin, gas: 1500000 });
    console.log('EventLogger contract deployed at:', eventLogger.options.address);
}

deploy().catch(console.error);

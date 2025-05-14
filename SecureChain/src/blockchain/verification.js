const Web3 = require('web3');
const fs = require('fs');
const EventLoggerABI = JSON.parse(fs.readFileSync('./contracts/EventLogger.json')).abi;

const web3 = new Web3('http://localhost:8545');  // Example for local node

async function verifyEvent(eventId) {
    const eventLoggerAddress = 'your_event_logger_contract_address';
    const eventLogger = new web3.eth.Contract(EventLoggerABI, eventLoggerAddress);

    const event = await eventLogger.methods.getEvent(eventId).call();
    console.log('Event details:', event);
}

verifyEvent('event_id_here').catch(console.error);

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title SecureChain Event Logger
 * @dev Smart contract for tamper-proof event logging from AI-driven event cameras
 * @author Rahul R. Nandan
 */
contract EventLogger {
    // Structs
    struct EventData {
        uint256 timestamp;
        string eventType;
        string location;
        uint8 confidenceScore;
        bytes32 dataHash;
        address deviceId;
        bool isAnomaly;
    }
    
    struct Device {
        bool isRegistered;
        string deviceName;
        string deviceLocation;
        address owner;
    }
    
    // State variables
    mapping(bytes32 => EventData) public events;
    mapping(address => Device) public registeredDevices;
    mapping(address => bool) public authorizedUsers;
    address public admin;
    uint256 public eventCount;
    
    // Events
    event AnomalyDetected(
        bytes32 indexed eventId,
        uint256 timestamp,
        string eventType,
        string location,
        uint8 confidenceScore,
        address deviceId
    );
    
    event NormalEventLogged(
        bytes32 indexed eventId,
        uint256 timestamp,
        string eventType
    );
    
    event DeviceRegistered(
        address indexed deviceId,
        string deviceName,
        string deviceLocation
    );
    
    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can call this function");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorizedUsers[msg.sender] || msg.sender == admin, "Not authorized");
        _;
    }
    
    modifier onlyRegisteredDevice() {
        require(registeredDevices[msg.sender].isRegistered, "Device not registered");
        _;
    }
    
    /**
     * @dev Constructor to set the contract admin
     */
    constructor() {
        admin = msg.sender;
        authorizedUsers[msg.sender] = true;
    }
    
    /**
     * @dev Register a new device that can log events
     * @param deviceId Address of the device
     * @param deviceName Name identifier for the device
     * @param deviceLocation Physical location of the device
     */
    function registerDevice(address deviceId, string memory deviceName, string memory deviceLocation) 
        external 
        onlyAuthorized 
    {
        require(!registeredDevices[deviceId].isRegistered, "Device already registered");
        
        registeredDevices[deviceId] = Device({
            isRegistered: true,
            deviceName: deviceName,
            deviceLocation: deviceLocation,
            owner: msg.sender
        });
        
        emit DeviceRegistered(deviceId, deviceName, deviceLocation);
    }
    
    /**
     * @dev Add a new authorized user
     * @param user Address of the user to authorize
     */
    function addAuthorizedUser(address user) external onlyAdmin {
        require(!authorizedUsers[user], "User already authorized");
        authorizedUsers[user] = true;
    }
    
    /**
     * @dev Remove an authorized user
     * @param user Address of the user to remove authorization
     */
    function removeAuthorizedUser(address user) external onlyAdmin {
        require(authorizedUsers[user], "User not authorized");
        require(user != admin, "Cannot remove admin");
        authorizedUsers[user] = false;
    }
    
    /**
     * @dev Log a detected event to the blockchain
     * @param eventType Type of event detected (e.g., "movement", "intrusion")
     * @param location Specific location within the monitored area
     * @param confidenceScore AI confidence score (0-100)
     * @param dataHash Hash of the original event data for verification
     * @param isAnomaly Whether the event is classified as an anomaly
     * @return eventId Unique identifier for the logged event
     */
    function logEvent(
        string memory eventType,
        string memory location,
        uint8 confidenceScore,
        bytes32 dataHash,
        bool isAnomaly
    ) 
        external 
        onlyRegisteredDevice 
        returns (bytes32 eventId) 
    {
        require(confidenceScore <= 100, "Confidence score must be between 0-100");
        
        // Generate a unique event ID
        eventId = keccak256(
            abi.encodePacked(
                block.timestamp,
                msg.sender,
                dataHash,
                eventCount
            )
        );
        
        // Store the event data
        events[eventId] = EventData({
            timestamp: block.timestamp,
            eventType: eventType,
            location: location,
            confidenceScore: confidenceScore,
            dataHash: dataHash,
            deviceId: msg.sender,
            isAnomaly: isAnomaly
        });
        
        // Increment the event counter
        eventCount++;
        
        // Emit appropriate event
        if (isAnomaly) {
            emit AnomalyDetected(
                eventId,
                block.timestamp,
                eventType,
                location,
                confidenceScore,
                msg.sender
            );
        } else {
            emit NormalEventLogged(
                eventId,
                block.timestamp,
                eventType
            );
        }
        
        return eventId;
    }
    
    /**
     * @dev Verify if an event matches the stored data
     * @param eventId ID of the event to verify
     * @param dataHash Hash of the data to verify against
     * @return isValid Whether the data matches
     */
    function verifyEvent(bytes32 eventId, bytes32 dataHash) 
        external 
        view 
        returns (bool isValid) 
    {
        return events[eventId].dataHash == dataHash;
    }
    
    /**
     * @dev Get detailed information about an event
     * @param eventId ID of the event to retrieve
     * @return Full event data structure
     */
    function getEventDetails(bytes32 eventId) 
        external 
        view 
        onlyAuthorized 
        returns (EventData memory) 
    {
        require(events[eventId].timestamp > 0, "Event does not exist");
        return events[eventId];
    }
    
    /**
     * @dev Check if a device is registered
     * @param deviceId Address of the device to check
     * @return isRegistered Whether the device is registered
     */
    function isDeviceRegistered(address deviceId) 
        external 
        view 
        returns (bool) 
    {
        return registeredDevices[deviceId].isRegistered;
    }
}

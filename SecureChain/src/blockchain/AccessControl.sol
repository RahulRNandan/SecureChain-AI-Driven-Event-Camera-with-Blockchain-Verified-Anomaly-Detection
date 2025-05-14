// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title AccessControl
 * @dev Contract managing access control for the SecureChain system
 */
contract AccessControl {
    // Owner of the contract
    address public owner;
    
    // Mapping of addresses to roles (0: none, 1: monitor, 2: admin)
    mapping(address => uint8) public roles;
    
    // Events
    event RoleAssigned(address indexed user, uint8 role);
    event RoleRevoked(address indexed user);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "AccessControl: caller is not the owner");
        _;
    }
    
    modifier onlyAdmin() {
        require(roles[msg.sender] == 2 || msg.sender == owner, "AccessControl: caller is not an admin");
        _;
    }
    
    modifier onlyMonitor() {
        require(roles[msg.sender] >= 1 || msg.sender == owner, "AccessControl: insufficient permissions");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Assigns a role to an address
     * @param user Address to assign the role to
     * @param role Role to assign (1: monitor, 2: admin)
     */
    function assignRole(address user, uint8 role) public onlyAdmin {
        require(role > 0 && role <= 2, "AccessControl: invalid role");
        roles[user] = role;
        emit RoleAssigned(user, role);
    }
    
    /**
     * @dev Revokes the role of an address
     * @param user Address to revoke the role from
     */
    function revokeRole(address user) public onlyAdmin {
        require(user != owner, "AccessControl: cannot revoke owner's role");
        delete roles[user];
        emit RoleRevoked(user);
    }
    
    /**
     * @dev Checks if an address has at least the required role
     * @param user Address to check
     * @param requiredRole Minimum role required
     * @return bool True if the address has the required role or higher
     */
    function hasRole(address user, uint8 requiredRole) public view returns (bool) {
        if (user == owner) return true;
        return roles[user] >= requiredRole;
    }
    
    /**
     * @dev Transfers ownership of the contract
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "AccessControl: new owner is the zero address");
        owner = newOwner;
    }
}

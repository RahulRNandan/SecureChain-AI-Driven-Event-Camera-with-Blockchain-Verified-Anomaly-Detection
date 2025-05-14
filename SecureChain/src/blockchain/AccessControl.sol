// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AccessControl {
    address public admin;
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Access denied. Only admin allowed.");
        _;
    }

    constructor() {
        admin = msg.sender;
    }

    function setAdmin(address newAdmin) public onlyAdmin {
        admin = newAdmin;
    }
}

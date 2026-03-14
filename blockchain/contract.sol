// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SkinDiseaseRecords {
    
    struct Record {
        uint256 userId;
        uint256 imageId;
        string sha256Hash;
        string ipfsCid;
        uint256 timestamp;
        address uploader;
    }

    mapping(uint256 => Record) public records; // Map imageId to Record

    // Event emitted on new record added
    event RecordAdded(uint256 indexed imageId, uint256 indexed userId, string sha256Hash);

    function addRecord(
        uint256 _userId,
        uint256 _imageId,
        string memory _sha256Hash,
        string memory _ipfsCid
    ) public {
        require(bytes(_sha256Hash).length > 0, "Hash cannot be empty");
        
        // Prevent duplicate imageIds
        require(records[_imageId].timestamp == 0, "Record for imageId already exists");

        records[_imageId] = Record({
            userId: _userId,
            imageId: _imageId,
            sha256Hash: _sha256Hash,
            ipfsCid: _ipfsCid,
            timestamp: block.timestamp,
            uploader: msg.sender
        });

        emit RecordAdded(_imageId, _userId, _sha256Hash);
    }

    function getRecord(uint256 _imageId) public view returns (
        uint256 userId,
        uint256 imageId,
        string memory sha256Hash,
        string memory ipfsCid,
        uint256 timestamp,
        address uploader
    ) {
        Record memory rec = records[_imageId];
        require(rec.timestamp != 0, "Record does not exist");
        
        return (
            rec.userId,
            rec.imageId,
            rec.sha256Hash,
            rec.ipfsCid,
            rec.timestamp,
            rec.uploader
        );
    }
}

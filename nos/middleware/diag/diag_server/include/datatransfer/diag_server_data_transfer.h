/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server data transfer
*/

#ifndef DIAG_SERVER_DATA_TRANSFER_H
#define DIAG_SERVER_DATA_TRANSFER_H

#include <mutex>
#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <string>
#include <dirent.h>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

namespace hozon {
namespace netaos {
namespace diag {

enum DataTransferType {
    NONE         = 0x00,
    UPLOAD       = 0x01,
    DOWNLOAD     = 0x02
};

enum DataTransferStatus {
    STANDBY,               // wait for start file transfer
    UPLOADING,             // file transfer upload in progress
    DOWNLOADING,           // file transfer download in progress
    COMPLETED              // file transfer completed
};

enum DataTransferSizeType {
    TRANSDATA,              // data size
    TRANSCAPACITY          // transmission capacity size
};

class DiagServerDataTransfer {

public:
    static DiagServerDataTransfer* getInstance();

    void Init();
    void DeInit();

    // start file data transfer
    bool StartFileDataUpload(const std::string& filePath);
    bool StartFileDataDownload(const std::string& filePath, const uint64_t& fileSize);

    // stop data transfer
    bool StopDataTransfer();

    // data transfer size
    uint64_t GetDataTransferSize() {return transfer_size_;}

    // data size
    uint64_t GetDataSize() {return data_size_;}

    // block sequence counter
    uint8_t GetBlockSequenceCounter() {return block_sequence_counter_;}

    uint16_t GetTotalBlockCount() {return total_block_count_;}
    uint16_t GetTransferBlockCount() {return transfer_block_count_;}

    // data transfer type
    DataTransferType GetDataTransferType() {return transfer_type_;}

    // data transfer status
    DataTransferStatus GetDataTransferStatus() {return transfer_status_;}

    // read data block by counter for upload
    bool ReadDataBlockByCounter(const uint8_t blockCounter, std::vector<uint8_t>& dataBlock);

    // write data to file by counter for download
    bool WriteDataToFileByCounter(const uint8_t blockCounter, const std::vector<uint8_t>& dataBlock);

    // size to vec with type
    void GetSizeToVecWithType(DataTransferSizeType type, std::vector<uint8_t>& sizeVec);

private:
    DiagServerDataTransfer();
    DiagServerDataTransfer(const DiagServerDataTransfer &);
    DiagServerDataTransfer & operator = (const DiagServerDataTransfer &);

    // stop file data transfer
    bool StopFileDataUpload();
    bool StopFileDataDownload();

    // files info for dir
    void GetFilesInfo();

private:
    static DiagServerDataTransfer* instance_;
    static std::mutex mtx_;

    // common
    DataTransferType transfer_type_;
    DataTransferStatus transfer_status_;
    uint64_t transfer_size_;
    uint64_t data_size_;
    uint8_t block_sequence_counter_;
    uint16_t total_block_count_;
    uint16_t transfer_block_count_;

    // download file
    std::string download_file_path_;
    std::ofstream download_ofs_;

    // upload file
    std::string upload_file_path_;
    uint64_t upload_data_offset_;
    std::vector<uint8_t> dir_info_;
    std::ifstream upload_ifs_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_DATA_TRANSFER_H

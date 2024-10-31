/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server data transfer
*/

#include <sys/stat.h>

#include "diag/diag_server/include/datatransfer/diag_server_data_transfer.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/info/diag_server_stored_info.h"

namespace hozon {
namespace netaos {
namespace diag {

const uint DISPLACEMENT = 8;
const std::string FILEEXTENSION = ".tmp";

DiagServerDataTransfer* DiagServerDataTransfer::instance_ = nullptr;
std::mutex DiagServerDataTransfer::mtx_;

DiagServerDataTransfer::DiagServerDataTransfer()
: transfer_type_(DataTransferType::NONE)
, transfer_status_(DataTransferStatus::STANDBY)
, transfer_size_(0)
, data_size_(0)
, block_sequence_counter_(0)
, total_block_count_(0)
, transfer_block_count_(0)
, download_file_path_("")
, upload_file_path_("")
, upload_data_offset_(0)
{

}

DiagServerDataTransfer*
DiagServerDataTransfer::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerDataTransfer();
        }
    }

    return instance_;
}

void
DiagServerDataTransfer::Init()
{
    DG_INFO << "DiagServerDataTransfer::Init";
}

void
DiagServerDataTransfer::DeInit()
{
    DG_INFO << "DiagServerDataTransfer::DeInit";
    StopDataTransfer();

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

bool
DiagServerDataTransfer::StartFileDataUpload(const std::string& filePath)
{
    DG_INFO << "DiagServerDataTransfer::StartFileDataUpload filePath: " << filePath;
    if (DataTransferStatus::STANDBY != transfer_status_) {
        DG_WARN << "DiagServerDataTransfer::StartFileDataUpload file transfer is already in progess.";
        return false;
    }

    if ("" == filePath){
        DG_ERROR << "DiagServerDataTransfer::StartFileDataUpload error input filePath: " << filePath;
        return false;
    }

    if (0 != access(filePath.c_str(), 0)) {
        DG_ERROR << "DiagServerDataTransfer::StartFileDataUpload filePath: " << filePath << " not exist!";
        return false;
    }

    transfer_type_ = DataTransferType::UPLOAD;
    transfer_size_ = DiagServerConfig::getInstance()->GetDiagServerDataTransferSize(transfer_type_);
    DG_INFO << "DiagServerDataTransfer::StartFileDataUpload upload data transfer_size_: " << transfer_size_;
    if (0 == transfer_size_) {
        DG_ERROR << "DiagServerDataTransfer::StartFileDataUpload server not allowed upload data. transfer_size_: " << transfer_size_;
        return false;
    }

    upload_file_path_ = filePath;
    struct stat statbuf;
    stat(upload_file_path_.c_str(), &statbuf);
    if(S_ISDIR(statbuf.st_mode)) {
        dir_info_.clear();
        GetFilesInfo();
        data_size_ = static_cast<uint64_t>(dir_info_.size());
    }
    else {
        data_size_ = static_cast<uint64_t>(statbuf.st_size);
    }

    if (0 == total_block_count_) {
        total_block_count_ = (data_size_ / (transfer_size_ -2)) + 1;
    }

    DG_INFO << "DiagServerDataTransfer::StartFileDataUpload data_size_: " << data_size_ << "Byte, total packages count: " << total_block_count_;
    transfer_status_ = DataTransferStatus::UPLOADING;
    return true;
}

bool
DiagServerDataTransfer::StopFileDataUpload()
{
    DG_INFO << "DiagServerDataTransfer::StopFileDataUpload";
    if (upload_ifs_.is_open()) {
        upload_ifs_.close();
    }

    transfer_type_ = DataTransferType::NONE;
    upload_file_path_ = "";
    upload_data_offset_ = 0;
    dir_info_.clear();
    return true;
}

bool
DiagServerDataTransfer::StartFileDataDownload(const std::string& filePath, const uint64_t& fileSize)
{
    DG_INFO << "DiagServerDataTransfer::StartFileDataDownload filePath: " << filePath << ", fileSize: " << fileSize;
    if (DataTransferStatus::STANDBY != transfer_status_) {
        DG_WARN << "DiagServerDataTransfer::StartFileDataDownload file transfer is already in progess.";
        return false;
    }

    if (("" == filePath) || (0 == fileSize)) {
        DG_ERROR << "DiagServerDataTransfer::StartFileDataDownload error input filePath: " << filePath << " fileSize" << fileSize;
        return false;
    }

    transfer_type_ = DataTransferType::DOWNLOAD;
    transfer_size_ = DiagServerConfig::getInstance()->GetDiagServerDataTransferSize(transfer_type_);
    DG_INFO << "DiagServerDataTransfer::StartFileDataDownload download data transfer_size_: " << transfer_size_;
    if (0 == transfer_size_) {
        DG_ERROR << "DiagServerDataTransfer::StartFileDataDownload server not allowed download data. transfer_size_: " << transfer_size_;
        return false;
    }

    download_file_path_ = filePath;
    data_size_ = fileSize;
    if (0 != access((download_file_path_ + FILEEXTENSION).c_str(), 0)) {
        std::ofstream ofs(download_file_path_ + FILEEXTENSION);
        ofs.close();
    }

    if (0 == total_block_count_) {
        total_block_count_ = (fileSize / (transfer_size_ -2)) + 1;
    }

    transfer_status_ = DataTransferStatus::DOWNLOADING;
    DG_INFO << "DiagServerDataTransfer::StartFileDataDownload total packages count: " << total_block_count_;
    return true;
}

bool
DiagServerDataTransfer::StopFileDataDownload()
{
    DG_INFO << "DiagServerDataTransfer::StopFileDataDownload";
    if (download_ofs_.is_open()) {
        download_ofs_.close();
    }

    transfer_type_ = DataTransferType::NONE;
    download_file_path_ = "";
    return true;
}

bool
DiagServerDataTransfer::StopDataTransfer()
{
    DG_INFO << "DiagServerDataTransfer::StopDataTransfer";
    transfer_status_ = DataTransferStatus::STANDBY;
    transfer_size_ = 0;
    data_size_ = 0;
    block_sequence_counter_ = 0;
    total_block_count_ = 0;
    transfer_block_count_ = 0;
    if (DataTransferType::UPLOAD == transfer_type_) {
        return StopFileDataUpload();
    }
    else if(DataTransferType::DOWNLOAD == transfer_type_) {
        return StopFileDataDownload();
    }
    else {
        return true;
    }
}

bool
DiagServerDataTransfer::ReadDataBlockByCounter(const uint8_t blockCounter, std::vector<uint8_t>& dataBlock)
{
    DG_DEBUG << "DiagServerDataTransfer::ReadDataBlockByCounter blockCounter: " << blockCounter;
    if (DataTransferStatus::UPLOADING != transfer_status_) {
        DG_ERROR << "DiagServerDataTransfer::ReadDataBlockByCounter error transfer_status_: " << transfer_status_;
        return false;
    }

    if (0 == data_size_) {
        DG_WARN << "DiagServerDataTransfer::ReadDataBlockByCounter null data_size_: " << data_size_;
        return false;
    }

    struct stat statbuf;
    stat(upload_file_path_.c_str(), &statbuf);
    if(!S_ISDIR(statbuf.st_mode)) {
        if (!upload_ifs_.is_open()) {
            upload_ifs_.open(upload_file_path_, std::ios::in | std::ios::binary);
            if (upload_ifs_.fail()) {
                DG_ERROR << "DiagServerDataTransfer::ReadDataBlockByCounter upload_file_path_: " << upload_file_path_ << " open failed.";
                return false;
            }
        }
    }

    uint64_t dataSize = data_size_;
    if (dataSize > (transfer_size_ - 2)) {
        dataSize = transfer_size_ - 2;
    }

    dataBlock.clear();
    dataBlock.resize(dataSize);
    if(S_ISDIR(statbuf.st_mode)) {
        dataBlock.assign(dir_info_.begin() + upload_data_offset_, dir_info_.begin() + upload_data_offset_ + dataSize);
    }
    else {
        char data[dataSize];
        upload_ifs_.seekg(upload_data_offset_, std::ios::beg);
        upload_ifs_.read(data, dataSize);
        dataBlock.assign(data, data + dataSize);
    }

    upload_data_offset_ += dataSize;
    data_size_ -= dataSize;

    if (data_size_ <= 0) {
        if (upload_ifs_.is_open()) {
            upload_ifs_.close();
        }

        transfer_status_ = DataTransferStatus::COMPLETED;
        upload_file_path_ = "";
        upload_data_offset_ = 0;
        dir_info_.clear();
    }

    DG_DEBUG << "DiagServerDataTransfer::ReadDataBlockByCounter dataBlock: " << UINT8_VEC_TO_STRING(dataBlock);
    block_sequence_counter_ = blockCounter;
    transfer_block_count_++;
    return true;
}

bool
DiagServerDataTransfer::WriteDataToFileByCounter(const uint8_t blockCounter, const std::vector<uint8_t>& dataBlock)
{
    DG_DEBUG << "DiagServerDataTransfer::WriteDataToFileByCounter blockCounter: " << blockCounter;
    if (DataTransferStatus::DOWNLOADING != transfer_status_) {
        DG_ERROR << "DiagServerDataTransfer::WriteDataToFileByCounter error transfer_status_: " << transfer_status_;
        return false;
    }

    if (0 == data_size_) {
        DG_WARN << "DiagServerDataTransfer::WriteDataToFileByCounter null data_size_: " << data_size_;
        return false;
    }

    if (!download_ofs_.is_open()) {
        download_ofs_.open(download_file_path_ + FILEEXTENSION, std::ios::out | std::ios::binary);
        if (download_ofs_.fail()) {
            DG_ERROR << "DiagServerDataTransfer::WriteDataToFileByCounter tmp download_file_path_: " << download_file_path_ + FILEEXTENSION << " open failed.";
            return false;
        }
    }

    DG_DEBUG << "DiagServerDataTransfer::WriteDataToFileByCounter dataBlock: " << UINT8_VEC_TO_STRING(dataBlock);
    std::string str = "";
    str.assign(dataBlock.begin(), dataBlock.end());
    download_ofs_ << str;
    download_ofs_.flush();
    data_size_ -= dataBlock.size();
    if (data_size_ <= 0) {
        if (download_ofs_.is_open()) {
            download_ofs_.close();
        }

        transfer_status_ = DataTransferStatus::COMPLETED;
        if (download_file_path_ == "/cfg/pki/oem_keys_preset.yaml.encrypted") {
            DG_INFO << "DiagServerDataTransfer::WriteDataToFileByCounter transfer /cfg/pki/oem_keys_preset.yaml.encrypted file completed!";
            DiagServerStoredInfo::getInstance()->KeysFileTransferCompletedToCFG();
        }
        int ret = rename((download_file_path_ + FILEEXTENSION).c_str(), download_file_path_.c_str());
        if (0 != ret) {
            DG_WARN << "DiagServerDataTransfer::WriteDataToFileByCounter download_file_path_ rename failed. code: " << ret;
        }

        download_file_path_ = "";
    }

    block_sequence_counter_ = blockCounter;
    transfer_block_count_++;
    return true;
}

void
DiagServerDataTransfer::GetFilesInfo()
{
    DG_INFO << "DiagServerDataTransfer::GetFilesInfo";
    DIR *dir;
    struct stat st;
    struct dirent *ent;
    if ((dir = opendir (upload_file_path_.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if(!strcmp(ent->d_name,".") || !strcmp(ent->d_name,"..")) {
                continue;
            }

            std::string name = upload_file_path_;
            stat((name.append("/").append(ent->d_name)).c_str(), &st);
            if(S_ISDIR(st.st_mode)) {
                dir_info_.emplace_back(0x01);
            }
            else if (S_ISREG(st.st_mode)) {
                dir_info_.emplace_back(0x00);
            }
            else {
                continue;
            }

            dir_info_.emplace_back(static_cast<uint8_t>(strlen(ent->d_name)));
            for(int i = 0; i < static_cast<uint8_t>(strlen(ent->d_name)); ++i) {
                dir_info_.emplace_back(static_cast<uint8_t>(ent->d_name[i]));
            }

            for (int i = 7; i >= 0; i--) {
                dir_info_.emplace_back(static_cast<uint8_t>(st.st_mtime >> (i * DISPLACEMENT)));
            }

            if (S_ISREG(st.st_mode)) {
                for (int i = 7; i >= 0; i--) {
                    dir_info_.emplace_back(static_cast<uint8_t>(st.st_size >> (i * DISPLACEMENT)));
                }
            }
        }

        closedir(dir);
    }
}

void
DiagServerDataTransfer::GetSizeToVecWithType(DataTransferSizeType type, std::vector<uint8_t>& sizeVec)
{
    uint64_t size = 0;
    if (DataTransferSizeType::TRANSCAPACITY == type) {
        size = static_cast<uint64_t>(transfer_size_);
    }
    else if (DataTransferSizeType::TRANSDATA == type) {
        size = static_cast<uint64_t>(data_size_);
    }
    else {
        return;
    }

    sizeVec.clear();
    uint displacement = (sizeof(size) / sizeof(uint8_t) - 1) * DISPLACEMENT;
    uint8_t iElement = 0;
    for (uint i = 0; i < (sizeof(size) / sizeof(uint8_t)); i++) {
        iElement = static_cast<uint8_t>((size >> displacement) & 0xff);
        if (0 != iElement) {
            sizeVec.push_back(iElement);
        }

        displacement -= DISPLACEMENT;
    }
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon

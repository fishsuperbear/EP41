/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: remote diag extension file transfer
*/

#include <sys/stat.h>
#include <fstream>
#include <thread>

#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/handler/remote_diag_handler.h"
#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/extension/remote_diag_file_transfer.h"
#include "remote_diag/include/common/md5.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

RemoteDiagFileTransfer* RemoteDiagFileTransfer::instance_ = nullptr;
std::mutex RemoteDiagFileTransfer::mtx_;

const std::string FILEEXTENSION = ".tmp";

RemoteDiagFileTransfer::RemoteDiagFileTransfer()
: stop_flag_(false)
, file_transfer_sa_("")
, file_transfer_ta_("")
, file_transfer_size_(1024)
, file_compress_dir_path_("./")
, file_download_dir_path_("./")
{
}

RemoteDiagFileTransfer*
RemoteDiagFileTransfer::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new RemoteDiagFileTransfer();
        }
    }

    return instance_;
}

void
RemoteDiagFileTransfer::Init()
{
    DGR_INFO << "RemoteDiagFileTransfer::Init";
    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    file_transfer_size_ = configInfo.FileTransferSize;
    file_compress_dir_path_ = configInfo.FileCompressDirPath;
    file_download_dir_path_ = configInfo.FileDownloadDirPath;
}

void
RemoteDiagFileTransfer::DeInit()
{
    DGR_INFO << "RemoteDiagFileTransfer::DeInit";
    stop_flag_ = true;
    if (REMOTE_DIAG_EXTENSION::REMOTE_DIAG_FILE_UPLOAD == RemoteDiagHandler::getInstance()->GetCurrExtension()) {
        Json::Value respMessage;
        respMessage["SA"] = file_transfer_sa_;
        respMessage["TA"] = file_transfer_ta_;
        respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kFileUpload];
        respMessage["DATA"] = "Program exit, file upload terminated!";
        RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
    }
    else if (REMOTE_DIAG_EXTENSION::REMOTE_DIAG_FILE_DOWNLOAD == RemoteDiagHandler::getInstance()->GetCurrExtension()) {
        Json::Value respMessage;
        respMessage["SA"] = file_transfer_sa_;
        respMessage["TA"] = file_transfer_ta_;
        respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kFileDownload];
        respMessage["BLOCK_SIZE"] = "0";
        respMessage["BLOCK_COUNT"] = "0";
        respMessage["DATA"] = "Program exit, file download terminated!";
        RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }

    DGR_INFO << "RemoteDiagFileTransfer::DeInit finish!";
}

std::string
RemoteDiagFileTransfer::FileDirCompress(const std::string& dirPath)
{
    DGR_INFO << "RemoteDiagFileTransfer::FileDirCompress dirPath: " << dirPath;
    if (access(dirPath.c_str(), F_OK) != 0) {
        return "";
    }

    auto vec = RemoteDiagHandler::Split(dirPath, "/");
    std::string dirName = vec[vec.size() - 1];
    std::string compressFilePath = file_compress_dir_path_ + dirName + ".zip";
    std::string command = "cd " + dirPath + "\n" + "zip -r " + compressFilePath + " ./*";
    int ret = system(command.c_str());
    if (0 != ret) {
        DGR_WARN << "RemoteDiagFileTransfer::FileDirCompress dirPath: " << dirPath << " compress failed.";
        return "";
    }

    return compressFilePath;
}

std::string
RemoteDiagFileTransfer::GetFileMD5(const std::string& filePath)
{
    DGR_INFO << "RemoteDiagFileTransfer::GetFileMD5 filePath: " << filePath;
    if (access(filePath.c_str(), F_OK) != 0) {
        DGR_WARN << "RemoteDiagFileTransfer::GetFileMD5 filePath: " << filePath << " not exist.";
        return "";
    }

    std::ifstream in(filePath, std::ios::binary);
	if (!in) {
        DGR_WARN << "RemoteDiagFileTransfer::GetFileMD5 filePath: " << filePath << " open failed.";
		return "";
	}

    MD5 md5;
    md5.reset();
	md5.update(in);
    std::string md5Value = md5.toString();
    DGR_INFO << "RemoteDiagFileTransfer::GetFileMD5 filePath: " << filePath << " MD5: " << md5Value;
    return md5Value;
}

void
RemoteDiagFileTransfer::FileUpload(const RemoteDiagFileUploadInfo& uploadInfo, bool isDirInfo)
{
    DGR_INFO << "RemoteDiagFileTransfer::FileUpload filepath: " << uploadInfo.uploadFilePath << " isDirInfo" << static_cast<uint>(isDirInfo);
    file_transfer_sa_ = uploadInfo.ta;
    file_transfer_ta_ = uploadInfo.sa;
    std::thread([this](const RemoteDiagFileUploadInfo& uploadInfo, bool isDirInfo) {
        std::string filepath = uploadInfo.uploadFilePath;
        if("DIR" == uploadInfo.uploadFileType) {
            filepath = FileDirCompress(filepath);
        }

        std::ifstream upload_ifs;
        upload_ifs.open(filepath, std::ios::in | std::ios::binary);
        Json::Value respMessage;
        respMessage["SA"] = file_transfer_sa_;
        respMessage["TA"] = file_transfer_ta_;
        if (isDirInfo) {
            respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kQueryDirInfo];
        }
        else {
            respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kFileDownload];
        }

        if(upload_ifs.fail()) {
            DGR_ERROR << "RemoteDiagFileTransfer::FileDownload open file: " << filepath << " failed! ";
            respMessage["BLOCK_SIZE"] = "0";
            respMessage["BLOCK_COUNT"] = "0";
            respMessage["DATA"] = "File download file open failed!";
            RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
            return;
        }

        RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_FILE_DOWNLOAD);
        RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(uploadInfo.sa.c_str(), 0, 16)), true);
        struct stat statbuf;
        stat(filepath.c_str(), &statbuf);
        uint32_t file_size = statbuf.st_size;
        uint32_t block_size = file_size / file_transfer_size_ + 1;
        respMessage["BLOCK_SIZE"] = std::to_string(block_size);
        respMessage["MD5_VALUE"] = GetFileMD5(filepath);

        uint32_t block_count = 0;
        uint32_t read_data_size = 0;
        uint32_t transfer_data_size = file_transfer_size_;
        while (read_data_size < file_size) {
            if (stop_flag_) {
                break;
            }

            if ((file_size - read_data_size) < file_transfer_size_) {
                transfer_data_size = file_size - read_data_size;
            }

            char data[transfer_data_size];
            upload_ifs.seekg(read_data_size, std::ios::beg);
            upload_ifs.read(data, transfer_data_size);
            // std::string data_str = "";
            // data_str.assign(data, data + transfer_data_size);
            std::string data_str = CHAR_ARRAY_TO_STRING_DATA(data, transfer_data_size);
            respMessage["DATA"] = data_str;
            respMessage["BLOCK_COUNT"] = std::to_string(++block_count);
            RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
            read_data_size += transfer_data_size;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        upload_ifs.close();
        if("DIR" == uploadInfo.uploadFileType) {
            remove(filepath.c_str());
        }

        RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_DEFAULT);
        RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(uploadInfo.sa.c_str(), 0, 16)), false);
        RemoteDiagHandler::getInstance()->FileDownloadCompleteCallback(uploadInfo.uploadFilePath);
    }, uploadInfo, isDirInfo).detach();


}

void
RemoteDiagFileTransfer::FileDownload(const RemoteDiagFileDownloadInfo& downloadInfo)
{
    DGR_INFO << "RemoteDiagFileTransfer::FileDownload dirpath: " << downloadInfo.downloadDirPath << " filename: " << downloadInfo.downloadFileName;
    file_transfer_sa_ = downloadInfo.ta;
    file_transfer_ta_ = downloadInfo.sa;
    std::string filepath = "";
    if ("" == downloadInfo.downloadDirPath) {
        filepath = file_download_dir_path_ + downloadInfo.downloadFileName;
    }
    else {
        filepath = downloadInfo.downloadDirPath + "/" + downloadInfo.downloadFileName;
    }

    if (0 != access((filepath + FILEEXTENSION).c_str(), 0)) {
        std::ofstream ofs(filepath + FILEEXTENSION);
        ofs.close();
    }

    std::ofstream download_ofs;
    if (1 == downloadInfo.blockCount) {
        download_ofs.open(filepath + FILEEXTENSION, std::ios::out | std::ios::binary);
    }
    else {
        download_ofs.open(filepath + FILEEXTENSION, std::ios::out | std::ios::binary | std::ios::app);
    }

    Json::Value respMessage;
    respMessage["SA"] = file_transfer_sa_;
    respMessage["TA"] = file_transfer_ta_;
    respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kFileUpload];
    if (download_ofs.fail()) {
        DGR_ERROR << "RemoteDiagFileTransfer::FileDownload tmp filepath: " << filepath + FILEEXTENSION << " open failed.";
        respMessage["DATA"] = "File upload file create failed!";
        RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
        return;
    }

    RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_FILE_UPLOAD);
    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(downloadInfo.sa.c_str(), 0, 16)), true);
    string data_str = "";
    for (uint i = 0; i < downloadInfo.data.size(); i++) {
        data_str += downloadInfo.data[i];
        if ((0 != i) && ((downloadInfo.data.size() - 1) != i)) {
            if (i % 2 != 0) {
                data_str += " ";
            }
        }
    }

    std::vector<uint8_t> vec;
    auto dataVec = RemoteDiagHandler::Split(data_str);
    for (auto& item : dataVec) {
        vec.emplace_back(static_cast<uint8_t>(std::strtoul(item.c_str(), 0, 16)));
    }

    std::string data = "";
    data.assign(vec.begin(), vec.end());
    download_ofs << data;
    download_ofs.flush();
    download_ofs.close();
    if (downloadInfo.blockCount >= downloadInfo.blockSize) {
        DGR_INFO << "RemoteDiagFileTransfer::FileDownload filename: " << downloadInfo.downloadFileName << " download complete!";
        if (downloadInfo.md5 == GetFileMD5(filepath + FILEEXTENSION)) {
            respMessage["DATA"] = "File uploaded successfully!";
        }
        else {
            respMessage["DATA"] = "Verification failed, MD5 is inconsistent!";
        }

        RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_DEFAULT);
        int ret = rename((filepath + FILEEXTENSION).c_str(), filepath.c_str());
        if (0 != ret) {
            DGR_WARN << "RemoteDiagFileTransfer::FileDownload filepath rename failed. code: " << ret;
        }

        RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
    }

    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(downloadInfo.sa.c_str(), 0, 16)), false);
}

void
RemoteDiagFileTransfer::QueryDirInfo(const RemoteDiagQueryDirInfo& dirInfo)
{
    DGR_INFO << "RemoteDiagFileTransfer::QueryDirInfo infoType: " << dirInfo.infoType << " dirPath: " << dirInfo.dirFilePath;
    std::string dirInfoPath = file_compress_dir_path_ + "dir_info.txt";
    std::string command = "";
    if ("Dir_info" == dirInfo.infoType) {
        command = "ls -lh " + dirInfo.dirFilePath + " > " + dirInfoPath;
    }
    else if ("Dir_tree_2" == dirInfo.infoType) {
        command = "tree -L 2 " + dirInfo.dirFilePath + " > " + dirInfoPath;
    }
    else if ("Dir_tree_3" == dirInfo.infoType) {
        command = "tree -L 3 " + dirInfo.dirFilePath + " > " + dirInfoPath;
    }
    else if ("Dir_tree_4" == dirInfo.infoType) {
        command = "tree -L 4 " + dirInfo.dirFilePath + " > " + dirInfoPath;
    }
    else if ("Dir_tree_all" == dirInfo.infoType) {
        command = "tree " + dirInfo.dirFilePath + " > " + dirInfoPath;
    }
    else {
        DGR_WARN << "RemoteDiagFileTransfer::QueryDirInfo invalid info type.";
        return;
    }

    if ("" != command)
    {
        int ret = system(command.c_str());
        if (0 != ret) {
            DGR_WARN << "RemoteDiagFileTransfer::QueryDirInfo dirInfoPath: " << dirInfoPath << " get failed.";
            return;
        }
    }

    RemoteDiagFileUploadInfo uploadInfo;
    uploadInfo.sa = dirInfo.sa;
    uploadInfo.ta = dirInfo.ta;
    uploadInfo.uploadFileType = "File";
    uploadInfo.uploadFilePath = dirInfoPath;
    FileUpload(uploadInfo, true);
}

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
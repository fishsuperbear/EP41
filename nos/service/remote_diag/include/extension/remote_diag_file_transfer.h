#ifndef REMOTE_DIAG_FILE_TRANSFER_H
#define REMOTE_DIAG_FILE_TRANSFER_H

#include <mutex>
#include <iostream>

#include "remote_diag/include/common/remote_diag_def.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

class RemoteDiagFileTransfer {

public:
    static RemoteDiagFileTransfer* getInstance();

    void Init();
    void DeInit();

    std::string FileDirCompress(const std::string& dirPath);
    std::string GetFileMD5(const std::string& filePath);

    void FileUpload(const RemoteDiagFileUploadInfo& uploadInfo, bool isDirInfo = false);
    void FileDownload(const RemoteDiagFileDownloadInfo& downloadInfo);
    void QueryDirInfo(const RemoteDiagQueryDirInfo& dirInfo);

private:
    RemoteDiagFileTransfer();
    RemoteDiagFileTransfer(const RemoteDiagFileTransfer &);
    RemoteDiagFileTransfer & operator = (const RemoteDiagFileTransfer &);

private:
    static RemoteDiagFileTransfer* instance_;
    static std::mutex mtx_;

    bool stop_flag_;

    std::string file_transfer_sa_;
    std::string file_transfer_ta_;

    uint32_t file_transfer_size_;
    std::string file_compress_dir_path_;
    std::string file_download_dir_path_;
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // #define REMOTE_DIAG_FILE_TRANSFER_H
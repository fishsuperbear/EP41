#include <sys/stat.h>
#include "phm_server/include/common/phm_server_def.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_utils.h"
#include "phm_server/include/fault_manager/file/phm_file_operate.h"

namespace hozon {
namespace netaos {
namespace phm_server {

FileOperate* FileOperate::instance_ = nullptr;
std::mutex FileOperate::mtx_;


FileOperate*
FileOperate::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new FileOperate();
        }
    }

    return instance_;
}

FileOperate::FileOperate()
{
}

void
FileOperate::Init()
{
    PHMS_INFO << "FileOperate::Init";
    if (!PHMUtils::MakeSurePath(fault_record_path)) {
        PHMS_ERROR << "FileOperate::Init MakeSurePath Error";
    }

    Backup(fault_record_file_A_path, fault_record_file_A_backup_path);
    Backup(fault_record_file_B_path, fault_record_file_B_backup_path);
    Delete(fault_record_file_A);
    Delete(fault_record_file_B);

    write_ofs_A_ = std::ofstream(fault_record_file_A_path, std::ios::out | std::ios::app);
    write_ofs_B_ = std::ofstream(fault_record_file_B_path, std::ios::out | std::ios::app);
    write_ofs_analysis_ = std::ofstream(analy_file_path, std::ios::out);
}

void
FileOperate::DeInit()
{
    PHMS_INFO << "FileOperate::DeInit";
    // TODO system status change
    Backup(system_running_time_file, system_running_time_file_backup);

    if (write_ofs_A_.is_open()) {
        write_ofs_A_.close();
    }

    if (write_ofs_B_.is_open()) {
        write_ofs_B_.close();
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

bool
FileOperate::Write(const std::string& file, const std::string& content)
{
    PHMS_INFO << "FileOperate::Write " << file;
    if (!PHMUtils::MakeSurePath(fault_record_path)) {
        PHMS_ERROR << "FileOperate::Write MakeSurePath Error";
        return false;
    }

    if (file == fault_record_file_A) {
        if (0 != access(fault_record_file_A_path.c_str(), F_OK)) {
            write_ofs_A_ = std::ofstream(fault_record_file_A_path, std::ios::out);
        }

        if (!write_ofs_A_.is_open()) {
            write_ofs_A_.open(fault_record_file_A_path, std::ios::out | std::ios::app);
            if (!write_ofs_A_.is_open()) {
                PHMS_ERROR << "FileOperate::Write file " << fault_record_file_A_path << " open failed!";
                return false;
            }
        }

        write_ofs_A_.write(content.c_str(), content.length());
        write_ofs_A_.flush();
    }
    else {
        if (0 != access(fault_record_file_B_path.c_str(), F_OK)) {
            write_ofs_B_ = std::ofstream(fault_record_file_B_path, std::ios::out);
        }

        if (!write_ofs_B_.is_open()) {
            write_ofs_B_.open(fault_record_file_B_path, std::ios::out | std::ios::app);
            if (!write_ofs_B_.is_open()) {
                PHMS_ERROR << "FileOperate::Write file " << fault_record_file_B_path << " open failed!";
                return false;
            }
        }

        write_ofs_B_.write(content.c_str(), content.length());
        write_ofs_B_.flush();
    }

    return true;
}

bool
FileOperate::WriteAnalysisFile(const std::string& analysisData)
{
    BackupAnalysisFile();

    if (0 != access(analy_file_path.c_str(), F_OK)) {
        write_ofs_analysis_ = std::ofstream(analy_file_path, std::ios::out);
    }

    if (!write_ofs_analysis_.is_open()) {
        write_ofs_analysis_.open(analy_file_path, std::ios::out);
        if (!write_ofs_analysis_.is_open()) {
            PHMS_ERROR << "FileOperate::WriteAnalysisFile open file failed!";
            return false;
        }
    }

    write_ofs_analysis_.write(analysisData.c_str(), analysisData.length());
    write_ofs_analysis_.flush();
    write_ofs_analysis_.close();
    return true;
}

void
FileOperate::BackupAnalysisFile()
{
    if (0 == access(analy_backup_file_path.c_str(), F_OK)) {
        std::remove(analy_backup_file_path.data());
    }

    if (0 == access(analy_file_path.c_str(), F_OK)) {
        // PHMS_INFO << "FileOperate::BackupAnalysisFile filePath:" << analy_file_path;
        struct stat statbuf;
        stat(analy_file_path.c_str(), &statbuf);
        size_t filesize = statbuf.st_size;
        if (filesize > 0) {
            PHMUtils::CopyFile(analy_file_path, analy_backup_file_path);
        }
    }

    return;
}

void
FileOperate::GetCollectData(std::vector<std::string>& otuAllFiles)
{
    PHMS_INFO << "FileOperate::GetCollectData";
    // backup file
    Backup(fault_record_file_A_path, fault_record_file_A_backup_path);
    Backup(fault_record_file_B_path, fault_record_file_B_backup_path);
    Backup(system_running_time_file, system_running_time_file_backup);

    const static std::string allFiles[] =
    {
        fault_record_file_A_path,
        fault_record_file_B_path,
        fault_record_file_A_backup_path,
        fault_record_file_B_backup_path,
        analy_file_path,
        analy_backup_file_path
    };

    // check file exsit & add
    for (auto& file : allFiles) {
        if (0 == access(file.c_str(), F_OK))
            otuAllFiles.emplace_back(file);
    }

    return;
}

bool
FileOperate::Backup(const std::string& src, const std::string& dst)
{
    PHMS_INFO << "FileOperate::Backup " << src;
    if (0 != access(src.c_str(), F_OK)) {
        PHMS_WARN << "FileOperate::Backup not have file filePath: " << src;
        return false;
    }

    struct stat statbuf;
    stat(src.c_str(), &statbuf);
    size_t filesize = statbuf.st_size;
    if (filesize <= 0) {
        PHMS_WARN << "FileOperate::Backup filePath: " << src << " size is 0.";
        return false;
    }

    PHMUtils::CopyFile(src, dst);
    return true;
}

bool
FileOperate::Delete(const std::string& file)
{
    PHMS_INFO << "FileOperate::Delete " << file;
    if (file == fault_record_file_A) {
        if (write_ofs_A_.is_open()) {
            write_ofs_A_.close();
        }

        PHMUtils::RemoveFile(fault_record_file_A_path);
    }
    else {
        if (write_ofs_B_.is_open()) {
            write_ofs_B_.close();
        }

        PHMUtils::RemoveFile(fault_record_file_B_path);
    }

    return true;
}

bool
FileOperate::Copy(const std::string& from, const std::string& to)
{
    return true;
}

bool
FileOperate::Sync(const std::string& file)
{
    if (file == fault_record_file_A) {
        if (write_ofs_A_.is_open()) {
            write_ofs_A_.flush();
        }
    }
    else {
        if (write_ofs_B_.is_open()) {
            write_ofs_B_.flush();
        }
    }

    return true;
}

bool
FileOperate::Sync()
{
    if (write_ofs_A_.is_open()) {
        write_ofs_A_.flush();
    }

    if (write_ofs_B_.is_open()) {
        write_ofs_B_.flush();
    }

    return true;
}

uint32_t
FileOperate::getSystemRunningTime(RUNNING_TIME_FILENAME fileType)
{
    std::string file = system_running_time_file;
    if (RUNNING_TIME_FILE_BACKUP == fileType) {
        file = system_running_time_file_backup;
    }

    if (0 != access(file.c_str(), 0)) {
        return 0;
    }

    std::ifstream fin;
    fin.open(file.c_str(), std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        PHMS_ERROR << "FileOperate::getSystemRunningTime read file: " << file << " open failed.";
        return 0;
    }

    std::string str = "";
    uint32_t runningTime = 0;
    while (fin >> str) {
        if ("time(seconds):" == str) {
            fin >> str;
            runningTime += static_cast<uint32_t>(strtoull(str.c_str(), 0, 10));
            break;
        }
    }

    fin.close();
    return runningTime;
}

void
FileOperate::RecordSystemRunningTime(const std::string& data)
{
    std::ofstream ofs;
    ofs.open(system_running_time_file, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
        PHMS_ERROR << "FileOperate::RecordSystemRunningTime write system_running_time_file open failed.";
        return;
    }

    ofs << data;
    ofs.close();
    return;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon

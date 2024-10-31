#ifndef PHM_FILE_OPERATE_H
#define PHM_FILE_OPERATE_H

#include <mutex>
#include <fstream>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace phm_server {

const std::string fault_record_path = "/opt/usr/col/fm";
const std::string fault_record_file_A = "hz_runtime_fault_list_A.txt";
const std::string fault_record_file_B = "hz_runtime_fault_list_B.txt";
const std::string fault_record_file_A_backup = "hz_runtime_fault_list_A_backup.txt";
const std::string fault_record_file_B_backup = "hz_runtime_fault_list_B_backup.txt";
const std::string fault_analy_file = "fault_analysis.txt";
const std::string fault_analy_backup_file = "fault_analysis_backup.txt";
const std::string fault_record_file_A_path = fault_record_path + "/" + fault_record_file_A;
const std::string fault_record_file_B_path = fault_record_path + "/" + fault_record_file_B;
const std::string fault_record_file_A_backup_path = fault_record_path + "/" + fault_record_file_A_backup;
const std::string fault_record_file_B_backup_path = fault_record_path + "/" + fault_record_file_B_backup;
const std::string analy_file_path = fault_record_path + "/" + fault_analy_file;
const std::string analy_backup_file_path = fault_record_path + "/" + fault_analy_backup_file;
const std::string system_running_time_file = "/opt/usr/col/rt/system_running_time.txt";
const std::string system_running_time_file_backup = "/cfg/rt/system_running_time_bk.txt";

enum RUNNING_TIME_FILENAME
{
    RUNNING_TIME_FILE = 0,
    RUNNING_TIME_FILE_BACKUP
};


class FileOperate {

public:
    static FileOperate* getInstance();

    void Init();
    void DeInit();

    bool Create(const std::string& file);
    bool Write(const std::string& file, const std::string& content);
    bool Backup(const std::string& src, const std::string& dst);
    bool Delete(const std::string& file);
    bool Copy(const std::string& from, const std::string& to);
    bool Sync(const std::string& file);
    bool Sync();
    bool WriteAnalysisFile(const std::string& analysisData);
    void BackupAnalysisFile();
    void GetCollectData(std::vector<std::string>& otuAllFiles);
    uint32_t getSystemRunningTime(RUNNING_TIME_FILENAME fileType);
    void RecordSystemRunningTime(const std::string& data);

private:
    FileOperate();
    FileOperate(const FileOperate &);
    FileOperate & operator = (const FileOperate &);

private:
    static std::mutex mtx_;
    static FileOperate* instance_;

    std::ofstream write_ofs_A_;
    std::ofstream write_ofs_B_;
    std::ofstream write_ofs_analysis_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_FILE_OPERATE_H

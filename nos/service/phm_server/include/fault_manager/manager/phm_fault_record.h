
#ifndef PHM_FAULT_RECORDER_H
#define PHM_FAULT_RECORDER_H

#include <mutex>
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {

class RecordFile {
public:
    RecordFile() {};
    RecordFile & operator = (const RecordFile & file)
    {
        this->file_name = file.file_name;
        this->file_size = file.file_size;
        return *this;
    }

public:
    std::string file_name;
    uint64_t file_size;
};

class FaultRecorder {

public:
    static FaultRecorder* getInstance();

    void Init();
    void DeInit();

    bool RefreshFaultFile();

private:
    FaultRecorder();
    FaultRecorder(const FaultRecorder &);
    FaultRecorder & operator = (const FaultRecorder &);

    void RecorderFaultCallback(Fault_t fault);
    std::string FaultFormat(const Fault_t& fault);
    void DeleteRecordFile(const std::string file);

private:
    static std::mutex mtx_;
    static std::mutex file_mtx_;
    static FaultRecorder* instance_;

    RecordFile latest_file_;
    uint64_t latest_file_size_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_FAULT_RECORDER_H
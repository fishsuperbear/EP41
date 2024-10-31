#ifndef FAULT_ANALYSIS_RECORD_H
#define FAULT_ANALYSIS_RECORD_H

#include <unordered_map>
#include <vector>
#include <mutex>
#include <fstream>
#include <chrono>
#include "phm_server/include/common/phm_server_def.h"


namespace hozon {
namespace netaos {
namespace phm_server {

class FaultAnalysisRecord {

public:
    static FaultAnalysisRecord* getInstance();
    void Init();
    void DeInit();

    void StartRecordAnalyDataToFile(std::string faultAnalyData);

private:
    FaultAnalysisRecord();
    FaultAnalysisRecord(const FaultAnalysisRecord &);
    FaultAnalysisRecord & operator = (const FaultAnalysisRecord &);

    void GetAnalysisRecordTime(std::string& content, uint8_t type);
    void RecordSystemRunningTime();

private:
    static FaultAnalysisRecord* instancePtr_;
    static std::mutex mtx_;
    std::ofstream write_ofs_;
    uint64_t frtst_time_;
    std::chrono::time_point<std::chrono::system_clock,
        std::chrono::system_clock::duration> startTime_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_ANALYSIS_RECORD_H

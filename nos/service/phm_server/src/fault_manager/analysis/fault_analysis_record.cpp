#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include "cfg/include/config_param.h"
#include "phm_server/include/common/phm_server_utils.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/file/phm_file_operate.h"
#include "phm_server/include/fault_manager/analysis/fault_analysis_record.h"

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace hozon::netaos::cfg;


FaultAnalysisRecord* FaultAnalysisRecord::instancePtr_ = nullptr;
std::mutex FaultAnalysisRecord::mtx_;

FaultAnalysisRecord::FaultAnalysisRecord()
: frtst_time_(0)
, startTime_(std::chrono::system_clock::now())
{
}

FaultAnalysisRecord*
FaultAnalysisRecord::getInstance()
{
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new FaultAnalysisRecord();
        }
    }

    return instancePtr_;
}

void
FaultAnalysisRecord::Init()
{
    PHMS_INFO << "FaultAnalysisRecord::Init enter!";
    frtst_time_ = PHMUtils::GetCurrentTime() / 1000000000;
    RecordSystemRunningTime();
}

void
FaultAnalysisRecord::DeInit()
{
    PHMS_INFO << "FaultAnalysisRecord::DeInit enter!";
    if (instancePtr_ != nullptr) {
        delete instancePtr_;
        instancePtr_ = nullptr;
    }
}

void
FaultAnalysisRecord::RecordSystemRunningTime()
{
    auto currentTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime_);
    uint32_t curRunningTime = static_cast<uint32_t>(double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    uint32_t lastRunTime = FileOperate::getInstance()->getSystemRunningTime(RUNNING_TIME_FILE);
    uint32_t lastRunTimeBack = FileOperate::getInstance()->getSystemRunningTime(RUNNING_TIME_FILE_BACKUP);
    uint32_t allRunningTime = curRunningTime;
    if (lastRunTimeBack > lastRunTime) {
        allRunningTime += lastRunTimeBack;
    }
    else {
        allRunningTime += lastRunTime;
    }

    std::string dataStr = "System running time(seconds): " + std::to_string(allRunningTime) + "\n";
    FileOperate::getInstance()->RecordSystemRunningTime(dataStr);
    startTime_ = currentTime;
    return;
}

void
FaultAnalysisRecord::StartRecordAnalyDataToFile(std::string faultAnalyData)
{
    PHMS_INFO << "FaultAnalysisRecord::StartRecordAnalyDataToFile";
    RecordSystemRunningTime();

    std::string str = "";
    std::string content = "";

    // FitstRecordTime
    GetAnalysisRecordTime(content, 0);
    str = "FitstRecordTime: " + content + "\n";

    // Version Info
    std::string sAllVersionInfo = "";
    std::string sVinInfo = "";
    if (nullptr != ConfigParam::Instance()) {
        ConfigParam::Instance()->GetParam("version/all", sAllVersionInfo);
        ConfigParam::Instance()->GetParam("dids/F190", sVinInfo);
    }
    str += "Version Infomation\n" + sAllVersionInfo + "\n";
    str += "VIN[" + sVinInfo + "]\n";

    // fault analy data
    str += faultAnalyData;

    // LastRecordTime
    GetAnalysisRecordTime(content, 1);
    str += "LastRecordTime: " + content + "\n";
    // PHMS_INFO << str.c_str();

    // backup & record file
    FileOperate::getInstance()->WriteAnalysisFile(str);
}

void
FaultAnalysisRecord::GetAnalysisRecordTime(std::string& content, uint8_t type)
{
    struct tm timeinfo = {0};
    char time_buf[128] = {0};
    time_t unix_time;
    if (0 == type) {
        unix_time = static_cast<time_t>(frtst_time_);
    }
    else {
        unix_time = static_cast<time_t>(PHMUtils::GetCurrentTime() / 1000000000);
    }

    localtime_r(&unix_time, &timeinfo);
    snprintf(time_buf, sizeof(time_buf) - 1, "%04d/%02d/%02d %02d:%02d:%02d UTC",
                                             timeinfo.tm_year + 1900,
                                             timeinfo.tm_mon + 1,
                                             timeinfo.tm_mday,
                                             timeinfo.tm_hour,
                                             timeinfo.tm_min,
                                             timeinfo.tm_sec);
    content = time_buf;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon

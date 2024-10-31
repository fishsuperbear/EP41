#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/file/phm_file_operate.h"
#include "phm_server/include/fault_manager/manager/phm_fault_record.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"
#include "phm_server/include/common/phm_server_config.h"

namespace hozon {
namespace netaos {
namespace phm_server {

FaultRecorder* FaultRecorder::instance_ = nullptr;
std::mutex FaultRecorder::mtx_;
std::mutex FaultRecorder::file_mtx_;

FaultRecorder*
FaultRecorder::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new FaultRecorder();
        }
    }

    return instance_;
}

FaultRecorder::FaultRecorder()
: latest_file_size_(1048576)
{
}

void
FaultRecorder::Init()
{
    PHMS_INFO << "FaultRecorder::Init";
    const PhmConfigInfo& configInfo = PHMServerConfig::getInstance()->GetPhmConfigInfo();
    latest_file_size_ = configInfo.CollectFaultFileSize;
    latest_file_.file_name = fault_record_file_A;
    latest_file_.file_size = 0U;
    FaultTaskHandler::getInstance()->RegisterRecorderCallback(std::bind(&FaultRecorder::RecorderFaultCallback, this, std::placeholders::_1));
}

void
FaultRecorder::DeInit()
{
    PHMS_INFO << "FaultRecorder::DeInit";
    FileOperate::getInstance()->Sync(fault_record_file_A);
    FileOperate::getInstance()->Sync(fault_record_file_B);

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
FaultRecorder::RecorderFaultCallback(Fault_t fault)
{
    uint32_t faultKey = fault.faultId * 100 + fault.faultObj;
    PHMS_INFO << "FaultRecorder::RecorderFaultCallback fault " << faultKey
              << " record: " << fault.faultAction.record;
    if (!fault.faultAction.record) {
        return;
    }

    if (PHMServerConfig::getInstance()->IsOverMaxRecordCount(faultKey)) {
        return;
    }
    PHMServerConfig::getInstance()->UpdateFaultRecordCount(faultKey);

    // PHMS_INFO << "FaultRecorder::RecorderFaultCallback file size " << latest_file_.file_size << " max size " << latest_file_size_;
    if (latest_file_.file_size > latest_file_size_) {
        if (latest_file_.file_name == fault_record_file_A) {
            FileOperate::getInstance()->Sync(fault_record_file_B);
            FileOperate::getInstance()->Delete(fault_record_file_B);
            latest_file_.file_name = fault_record_file_B;
        }
        else if (latest_file_.file_name == fault_record_file_B) {
            FileOperate::getInstance()->Sync(fault_record_file_A);
            FileOperate::getInstance()->Delete(fault_record_file_A);
            latest_file_.file_name = fault_record_file_A;
        }

        latest_file_.file_size = 0U;
    }

    std::string fault_str = FaultFormat(fault);
    latest_file_.file_size += fault_str.length();
    PHMS_INFO << "FaultRecorder::RecorderFaultCallback cur file:" << latest_file_.file_name
              << " fileSize: " << latest_file_.file_size;
    FileOperate::getInstance()->Write(latest_file_.file_name, fault_str);
}

void
FaultRecorder::DeleteRecordFile(const std::string file)
{
    PHMS_INFO << "FaultRecorder::DeleteRecordFile file:" << file;
    FileOperate::getInstance()->Delete(file);
    FileOperate::getInstance()->Sync(file);
    latest_file_.file_size = 0;
    latest_file_.file_name = file;
}

std::string
FaultRecorder::FaultFormat(const Fault_t& fault)
{
    char time_format[50] = {0};
    uint64_t time_sec = fault.faultOccurTime / 1000000000;
    uint64_t time_nsec = fault.faultOccurTime % 1000000000;
    std::stringstream nsec_format;
    nsec_format << std::setw(9) << std::setfill('0') << time_nsec;
    time_t gmt = static_cast<time_t>(time_sec);
    strftime(time_format, sizeof time_format, "%D %T", gmtime(&gmt));

    std::string fault_str = "OccurTime:" + std::string(time_format) + "." + nsec_format.str() + " UTC    " +
                            "Domain[" + fault.faultDomain + "] Fault[" + std::to_string(fault.faultId*100 + fault.faultObj) + "] " +
                            "Status[" + std::to_string(fault.faultStatus) +"] Process[" + fault.faultProcess + "] " +
                            "Describe[" + fault.faultDscribe + "]\n";

    return fault_str;
}

bool
FaultRecorder::RefreshFaultFile()
{
    PHMS_INFO << "FaultRecorder::RefreshFaultFile";
    FileOperate::getInstance()->Sync(fault_record_file_A);
    FileOperate::getInstance()->Sync(fault_record_file_B);
    return true;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon

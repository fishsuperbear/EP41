#include <regex>
#include <fstream>
#include "system_monitor/include/handler/system_monitor_handler.h"
#include "system_monitor/include/manager/system_monitor_manager.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

SystemMonitorHandler* SystemMonitorHandler::instance_ = nullptr;
std::mutex SystemMonitorHandler::mtx_;

const std::string DIDS_DATA_FILE_PATH = "/cfg/dids/dids.json";
const std::string DIDS_DATA_BACK_FILE_PATH = "/cfg/dids/dids.json_bak_1";

SystemMonitorHandler::SystemMonitorHandler()
: phm_client_(new PHMClient())
, event_receiver_(new SystemMonitorTransportEventReceiver())
{
}

SystemMonitorHandler*
SystemMonitorHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new SystemMonitorHandler();
        }
    }

    return instance_;
}

void
SystemMonitorHandler::Init()
{
    STMM_INFO << "SystemMonitorHandler::Init";
    // get vin number
    std::string vin = GetVinNumber();

    // phm client init
    if (nullptr != phm_client_) {
        phm_client_->Init("", nullptr, nullptr, "system_monitorProcess");
    }

    // event receiver init
    if (nullptr != event_receiver_) {
        event_receiver_->Init(vin);
    }
}

void
SystemMonitorHandler::DeInit()
{
    STMM_INFO << "SystemMonitorHandler::DeInit";
    // event receiver deinit
    if (nullptr != event_receiver_) {
        event_receiver_->DeInit();
        delete event_receiver_;
        event_receiver_ = nullptr;
    }

    // phm client deinit
    if (nullptr != phm_client_) {
        phm_client_->Deinit();
        delete phm_client_;
        phm_client_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
SystemMonitorHandler::ControlEventCallBack(const SystemMonitorControlEventInfo& info)
{
    STMM_INFO << "SystemMonitorHandler::ControlEventCallBack monitor_id: " << info.id
                                                      << " control_type: " << info.type
                                                      << " control_value: " << info.value;
    std::lock_guard<std::mutex> lck(mtx_);
    SystemMonitorManager::getInstance()->ControlEvent(info);
}

void
SystemMonitorHandler::RefreshEventCallback(const std::string& reason)
{
    STMM_INFO << "SystemMonitorHandler::RefreshEventCallback refresh_reason: " << reason;
    std::lock_guard<std::mutex> lck(mtx_);
    SystemMonitorManager::getInstance()->RefreshEvent(reason);
}

bool
SystemMonitorHandler::ReportFault(const SystemMonitorSendFaultInfo& faultInfo)
{
    STMM_INFO << "SystemMonitorHandler::ReportFault faultId: " << faultInfo.faultId
                                              << " faultObj: " << faultInfo.faultObj
                                              << " faultStatus: " << faultInfo.faultStatus;
    std::lock_guard<std::mutex> lck(mtx_);
    if (nullptr == phm_client_) {
        STMM_ERROR << "SystemMonitorHandler::ReportFault error: phm_client_ is nullptr!";
        return false;
    }

    SendFault_t fault(faultInfo.faultId, faultInfo.faultObj, faultInfo.faultStatus);
    int32_t result = phm_client_->ReportFault(fault);
    if (result < 0) {
        STMM_WARN << "SystemMonitorHandler::ReportFault failed. failedCode: " << result;
        return false;
    }

    return true;
}

std::string
SystemMonitorHandler::GetVinNumber()
{
    std::string filePath = "";
    if (0 == access(DIDS_DATA_FILE_PATH.c_str(), F_OK)) {
        filePath = DIDS_DATA_FILE_PATH;
    }
    else {
        if (0 == access(DIDS_DATA_BACK_FILE_PATH.c_str(), F_OK)) {
            filePath = DIDS_DATA_BACK_FILE_PATH;
        }
    }

    if ("" == filePath) {
        return "";
    }

    std::ifstream ifs;
    ifs.open(filePath, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        return "";
    }

    std::string vin = "";
    std::string str = "";
    bool bFind = false;
    while (getline(ifs, str))
    {
        if (std::string::npos != str.find("F190")) {
            bFind = true;
            continue;
        }

        if (bFind && (std::string::npos != str.find("string"))) {
            auto vec = Split(str, "\"");
            if (vec.size() > 3) {
                vin = vec[3];
            }

            break;
        }
    }

    ifs.close();
    return vin;
}

std::vector<std::string>
SystemMonitorHandler::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
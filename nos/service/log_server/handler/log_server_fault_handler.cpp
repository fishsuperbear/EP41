#include "log_server/handler/log_server_fault_handler.h"
#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {

logServerFaultHandler* logServerFaultHandler::instance_ = nullptr;
std::mutex logServerFaultHandler::mtx_;

logServerFaultHandler*
logServerFaultHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new logServerFaultHandler();
        }
    }

    return instance_;
}

logServerFaultHandler::logServerFaultHandler()
 : phm_client_{std::make_unique<PHMClient>()}
{
}

void
logServerFaultHandler::Init()
{
    LOG_SERVER_INFO << "logServerFaultHandler::Init.";
    if (nullptr != phm_client_) {
        phm_client_->Init();
        phm_client_->Start();
    }
    LOG_SERVER_INFO << "logServerFaultHandler::Init Done.";
}

void
logServerFaultHandler::DeInit()
{
    LOG_SERVER_INFO << "logServerFaultHandler::DeInit.";
    if (nullptr != phm_client_) {
        phm_client_->Deinit();
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
    LOG_SERVER_INFO << "logServerFaultHandler::DeInit Done.";
}

bool
logServerFaultHandler::ReportFault(const LogServerSendFaultInfo& faultInfo)
{
    LOG_SERVER_DEBUG << "logServerFaultHandler::ReportFault faultId: " << faultInfo.faultId
                                              << " faultObj: " << faultInfo.faultObj
                                              << " faultStatus: " << faultInfo.faultStatus;
    std::lock_guard<std::mutex> lck(mtx_);
    if (nullptr == phm_client_) {
        LOG_SERVER_ERROR << "logServerFaultHandler::ReportFault error: phm_client_ is nullptr!";
        return false;
    }

    SendFault_t fault(faultInfo.faultId, faultInfo.faultObj, faultInfo.faultStatus);
    int32_t result = phm_client_->ReportFault(fault);
    if (result < 0) {
        LOG_SERVER_WARN << "logServerFaultHandler::ReportFault failed. failedCode: " << result;
        return false;
    }

    return true;
}

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

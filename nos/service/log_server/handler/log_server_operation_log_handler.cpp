#include "log_server/handler/log_server_operation_log_handler.h"
#include "log_server/handler/log_server_fault_handler.h"
#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {

logServerOperationLogHandler* logServerOperationLogHandler::instance_ = nullptr;
std::mutex logServerOperationLogHandler::mtx_;

logServerOperationLogHandler*
logServerOperationLogHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new logServerOperationLogHandler();
        }
    }

    return instance_;
}

logServerOperationLogHandler::logServerOperationLogHandler()
:impl_(std::make_unique<OperationLogImpl>())
{
}

void
logServerOperationLogHandler::Init()
{
    LOG_SERVER_INFO << "logServerOperationLogHandler::Init.";
    auto res = impl_->Init();
    if (res != 0)
    {
        LogServerSendFaultInfo info{};
        info.faultId = 4200;
        info.faultObj = 1;
        info.faultStatus = 1;
        logServerFaultHandler::getInstance()->ReportFault(info);
        LOG_SERVER_ERROR << "logServerOperationLogHandler::Init error!";
    }
    LOG_SERVER_INFO << "logServerOperationLogHandler::Init Done.";
}

void
logServerOperationLogHandler::DeInit()
{
    LOG_SERVER_INFO << "logServerOperationLogHandler::DeInit.";
    auto res = impl_->DeInit();
    if (res != 0)
    {
        LOG_SERVER_ERROR << "logServerOperationLogHandler::DeInit error!";
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
    LOG_SERVER_INFO << "logServerOperationLogHandler::DeInit Done.";
}


}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

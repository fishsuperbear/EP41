#include "log_server/handler/log_server_compress_handler.h"
#include "log_server/handler/log_server_fault_handler.h"
#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {

logServerCompressHandler* logServerCompressHandler::instance_ = nullptr;
std::mutex logServerCompressHandler::mtx_;

logServerCompressHandler*
logServerCompressHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new logServerCompressHandler();
        }
    }

    return instance_;
}

logServerCompressHandler::logServerCompressHandler()
:impl_(std::make_unique<CompressLogImpl>())
{
}

void
logServerCompressHandler::Init()
{
    LOG_SERVER_INFO << "logServerOperationLogHandler::Init.";
    auto res = impl_->Init();
    if (res != 0)
    {
        LogServerSendFaultInfo info{};
        info.faultId = 4200;
        info.faultObj = 2;
        info.faultStatus = 1;
        logServerFaultHandler::getInstance()->ReportFault(info);
        LOG_SERVER_ERROR << "logServerCompressHandler::Init error!";
    }
    LOG_SERVER_INFO << "logServerOperationLogHandler::Init Done.";
}

void
logServerCompressHandler::DeInit()
{
    LOG_SERVER_INFO << "logServerCompressHandler::DeInit.";
    auto res = impl_->DeInit();
    if (res != 0)
    {
        LOG_SERVER_ERROR << "logServerCompressHandler::DeInit error!";
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
    LOG_SERVER_INFO << "logServerCompressHandler::DeInit Done.";
}


}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

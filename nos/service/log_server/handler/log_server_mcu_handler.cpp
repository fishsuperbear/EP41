#include "log_server/handler/log_server_mcu_handler.h"
#include "log_server/handler/log_server_fault_handler.h"
#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {

logServerMcuHandler* logServerMcuHandler::instance_ = nullptr;
std::mutex logServerMcuHandler::mtx_;

logServerMcuHandler*
logServerMcuHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new logServerMcuHandler();
        }
    }

    return instance_;
}

logServerMcuHandler::logServerMcuHandler()
:udp_msg_impl_(std::make_unique<UdpMsgImpl>()),
mcu_log_impl_(std::make_unique<McuLogImpl>())
{
}

void
logServerMcuHandler::Init()
{
    LOG_SERVER_INFO << "logServerMcuHandler::Init.";
    auto res = udp_msg_impl_->Init(mcu_log_ip, mcu_log_port);
    if (res != 0)
    {
        LogServerSendFaultInfo info{};
        info.faultId = 4200;
        info.faultObj = 4;
        info.faultStatus = 1;
        logServerFaultHandler::getInstance()->ReportFault(info);
        LOG_SERVER_ERROR << "logServerMcuHandler::Init udp_msg_impl_ error, error code is : " << res;
    }
    udp_msg_impl_->SetMcuLogCallback(std::bind(&logServerMcuHandler::ReceiveMcuLog, this, std::placeholders::_1));
    udp_msg_impl_->Start();

    res = mcu_log_impl_->Init();
    if (res != 0)
    {
        LogServerSendFaultInfo info{};
        info.faultId = 4200;
        info.faultObj = 3;
        info.faultStatus = 1;
        logServerFaultHandler::getInstance()->ReportFault(info);
        LOG_SERVER_ERROR << "logServerMcuHandler::Init mcu_log_impl_ error, error code is : " << res;
    }
    LOG_SERVER_INFO << "logServerMcuHandler::Init Done.";
}

void
logServerMcuHandler::DeInit()
{
    LOG_SERVER_INFO << "logServerMcuHandler::DeInit.";
    udp_msg_impl_->Stop();
    auto res = udp_msg_impl_->DeInit();
    if (res != 0)
    {
        LOG_SERVER_ERROR << "logServerMcuHandler::DeInit udp_msg_impl_ error, error code is : " << res;
    }

    res = mcu_log_impl_->DeInit();
    if (res != 0)
    {
        LOG_SERVER_ERROR << "logServerMcuHandler::DeInit mcu_log_impl_ error, error code is : " << res;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
    LOG_SERVER_INFO << "logServerMcuHandler::DeInit Done.";
}

void 
logServerMcuHandler::ReceiveMcuLog(const McuLog& mcuLog)
{
    LOG_SERVER_DEBUG << "logServerMcuHandler::ReceiveMcuLog";
    auto res = mcu_log_impl_->LogOut(mcuLog);
    if (res != 0)
    {
        LOG_SERVER_ERROR << "logout error. code is : " << res;
    }
}


}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

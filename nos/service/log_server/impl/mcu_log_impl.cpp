#include "log_server/impl/mcu_log_impl.h"
#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {

McuLogImpl::McuLogImpl()
:mcu_log_()
{
}

int32_t
McuLogImpl::Init()
{
    LOG_SERVER_INFO << "McuLogImpl::Init";
    LOG_SERVER_INFO << "McuLogImpl::Init Done";
    return 0;
}

int32_t
McuLogImpl::DeInit()
{
    LOG_SERVER_INFO << "McuLogImpl::DeInit";
    LOG_SERVER_INFO << "McuLogImpl::DeInit Done";
    return 0;
}
int32_t 
McuLogImpl::LogOut(const McuLog& mcuLog)
{
    mcu_log_ = mcuLog;
    std::string appId{};
    std::string ctxId{};
    uint16_t level{0x00};
    std::string msg{};
    GetAppId(appId);
    GetCtxId(ctxId);
    GetLogLevel(level);
    GetLogMsg(msg);
    return InitLog(appId, ctxId, level, msg);
}

int32_t 
McuLogImpl::InitLog(const std::string appId, const std::string ctxId, const uint16_t ctxLogLevel, const std::string& msg)
{
    hozon::netaos::log::InitMcuLogging(appId);
    auto log = hozon::netaos::log::CreateMcuLogger(appId, ctxId);
    switch (ctxLogLevel)
    {
    case 0x00:
        log->LogTrace() << msg;
        break;
    case 0x01:
        log->LogDebug() << msg;
        break;
    case 0x02:
        log->LogInfo() << msg;
        break;
    case 0x03:
        log->LogWarn() << msg;
        break;
    case 0x04:
        log->LogError() << msg;
        break;
    case 0x05:
        log->LogCritical() << msg;
        break;
    case 0x06:
        break;
    default:
        LOG_SERVER_DEBUG << "logout error, Please check log level!";
        return -1;
        break;
    }
    return 0;
}

int32_t 
McuLogImpl::GetLogMsg(std::string& msg)
{
    // TODO
    std::string sec = std::to_string(mcu_log_.header.stamp.sec);
    std::string nsec = std::to_string(mcu_log_.header.stamp.nsec);
    std::string data{mcu_log_.log.data(), mcu_log_.log.data() + mcu_log_.header.length};

    msg = sec + ":" + nsec + "] [" + data;
    LOG_SERVER_DEBUG << "final msg is : " << msg;
    return 0;
}

void 
McuLogImpl::GetAppId(std::string& appid)
{
    if (mcu_log_.header.app_id == 0x01) {
        appid = "DESAY_MCU";
    } else if (mcu_log_.header.app_id == 0x02) {
        appid = "HZ_MCU";
    } else {
        appid = "";
    }
}
void 
McuLogImpl::GetCtxId(std::string& ctxid)
{
    if (mcu_log_.header.ctx_id == 0x00) {
        ctxid = "MCU";
    } else if (mcu_log_.header.ctx_id == 0x01) {
        ctxid = "CAN";
    } else if (mcu_log_.header.ctx_id == 0x02) {
        ctxid = "ETH";
    } else if (mcu_log_.header.ctx_id == 0x03) {
        ctxid = "PwrM";
    } else {
        ctxid = "DEFAULT";
    }
}

void 
McuLogImpl::GetLogLevel(uint16_t& level)
{
    level = mcu_log_.header.level;
}

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

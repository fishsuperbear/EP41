#pragma once

#include <string>
#include "log_server/common/log_server_def.h"

namespace hozon {
namespace netaos {
namespace logserver {

class McuLogImpl
{

public:
    McuLogImpl();
    virtual ~McuLogImpl(){};
    int32_t Init();
    int32_t DeInit();

    int32_t LogOut(const McuLog& mcuLog);
private:
    int32_t InitLog(const std::string appId, const std::string ctxId, const uint16_t ctxLogLevel, const std::string& msg);
    int32_t GetLogMsg(std::string& msg);
    void GetAppId(std::string& appid);
    void GetCtxId(std::string& ctxid);
    void GetLogLevel(uint16_t& level);
private:
    McuLog mcu_log_;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

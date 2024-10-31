#pragma once
#include <functional>
#include <vector>
#include <thread>

#include "log/include/logging.h"
#include "log_server/common/log_server_def.h"

namespace hozon {
namespace netaos {
namespace logserver {

class UdpMsgImpl
{
public:
    using McuLogCallback = std::function<void(const McuLog &)>;
    UdpMsgImpl();
    virtual ~UdpMsgImpl(){};
    int32_t Init(const std::string& ip, const uint32_t& port);
    int32_t DeInit();

    void SetMcuLogCallback(McuLogCallback callback);
    int32_t Start();
    int32_t Stop();

    int32_t WaitRequest();

private:
    bool ParseLog(const std::vector<uint8_t>& udpData, McuLog& log);

private:
    int socket_fd_;
    std::thread waitReq_;
    McuLogCallback mcu_log_callback_;
    std::atomic<bool> is_connected_ {false};
    std::atomic<bool> is_quit_ {false};
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

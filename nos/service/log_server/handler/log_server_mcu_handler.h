#pragma once

#include <mutex>
#include <functional>
#include <memory>

#include "log_server/impl/mcu_log_impl.h"
#include "log_server/impl/udp_msg_impl.h"

namespace hozon {
namespace netaos {
namespace logserver {

class logServerMcuHandler {

public:
    static logServerMcuHandler* getInstance();

    void Init();
    void DeInit();

    void ReceiveMcuLog(const McuLog& mcuLog);
private:
    logServerMcuHandler();
    logServerMcuHandler(const logServerMcuHandler &);
    logServerMcuHandler & operator = (const logServerMcuHandler &);

private:
    static std::mutex mtx_;
    static logServerMcuHandler* instance_;
    
    std::unique_ptr<UdpMsgImpl> udp_msg_impl_;
    std::unique_ptr<McuLogImpl> mcu_log_impl_;

};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

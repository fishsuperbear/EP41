#pragma once

#include <functional>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

class McuUdsPeri
{
public:
    using McuUdsCallback = std::function<void(const McuUdsMsg &)>;

    static McuUdsPeri* Instance();

    int32_t Init(const std::string& ip, const uint32_t& port);
    int32_t DeInit();

    void SetMcuUdsCallback(McuUdsCallback callback);
    int32_t Start();
    int32_t Stop();

    int32_t WaitRequest();
    
private:
    McuUdsPeri();
    ~McuUdsPeri();
    McuUdsPeri(const McuUdsPeri &);
    McuUdsPeri & operator = (const McuUdsPeri &);

    bool ParseMsg(const std::vector<uint8_t>& udpData, McuUdsMsg& msg);

private:
    static std::mutex m_mtx;
    static McuUdsPeri* m_pInstance;

    int socket_fd_;
    std::thread waitReq_;
    McuUdsCallback mcu_uds_callback_;
    std::atomic<bool> is_connected_ {false};
    std::atomic<bool> is_quit_ {false};
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon

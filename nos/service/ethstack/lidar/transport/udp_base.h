#ifndef LIDAR_TRANSPORT_UDP_BASE_H
#define LIDAR_TRANSPORT_UDP_BASE_H

#include <mutex>
#include <list>
#include <thread>
#include <condition_variable>

#include "common/lidar_types.h"
#include "udp_dispatcher.h"
#include "faultmessage/lidar_fault_report.h"
#include "cfg/include/config_param.h"
#include "faultmessage/lidar_status_report.h"

namespace hozon {
namespace ethstack {
namespace lidar {

class UdpBase
{
public:
    UdpBase(EthernetSocketInfo socket);
    virtual ~UdpBase();

    int32_t Init();
    int32_t Start();
    int32_t StartProcess();
    int32_t Stop();

    int32_t RecvEthernetPacket(EthernetPacket* packet);
    int32_t SendEthernetPacket(const EthernetPacket* packet);

    bool IsStop() const;

    int32_t GetSocketFd() const;

    std::list<std::shared_ptr<EthernetPacket>>& GetRecvQueue();

private:
    UdpBase(const UdpBase&);
    UdpBase& operator=(const UdpBase&);

    int32_t SetupEthernetSocket(const EthernetSocketInfo& socket);

    int32_t PollThread();
    int32_t ProcessThread();

private:
    int32_t m_fd = -1;
    bool stopFlag_ = false;

    EthernetSocketInfo m_socketInfo;

    std::mutex send_queue_mutex_;
    std::mutex recv_queue_mutex_;

    std::list<std::shared_ptr<EthernetPacket>> send_queue_;
    std::list<std::shared_ptr<EthernetPacket>> recv_queue_;

    std::thread poll_thread_;
    std::thread process_thread_;

    std::condition_variable poll_condition_;
    std::condition_variable process_condition_;

    std::shared_ptr<UdpDispatcher> m_dispatcher;
};


}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // LIDAR_TRANSPORT_UDP_BASE_H

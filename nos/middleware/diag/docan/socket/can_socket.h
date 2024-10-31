/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CanSocket Header
 */


#ifndef DOCAN_SOCKET_H_
#define DOCAN_SOCKET_H_

#include <mutex>
#include <list>
#include <thread>
#include <condition_variable>
#include <linux/can.h>
#include <linux/can/raw.h>
#include "diag/docan/common/docan_internal_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class PacketParser;
class CanSocket
{
public:
    CanSocket(N_EcuInfo_t ecuInfo);
    virtual ~CanSocket();

    int32_t Init();
    int32_t Start();
    int32_t Stop();

    N_EcuInfo_t GetEcuInfo();

    static bool GetIfFlags(const std::string& ifname);
    static int32_t SetIfFlags(const std::string& ifname, bool up);

    bool IsStop();

    int32_t AddSendPacket(const CanPacket& packet);
    int32_t AddSendQueue(const std::vector<CanPacket>& queue);

    int32_t SetRecvTimeout(const struct timeval &tv) const;
    int32_t SetCanFilters(const std::vector<can_filter> &filters);

    int32_t SetPacketParser(std::shared_ptr<PacketParser> parser);

    std::list<CanPacket>& GetRecvQueue();

private:
    CanSocket(const CanSocket&);
    CanSocket& operator=(const CanSocket&);

    int32_t RecvCanFrame(CanPacket& packet);
    int32_t SendCanFrame(const CanPacket& packet);
    int32_t SetupCanSocket(const N_EcuInfo_t& ecuInfo);

    int32_t WriteThread();
    int32_t ReadThread();
    int32_t ProcessThread();

private:
    int32_t m_fd = -1;
    bool stopFlag_ = false;

    N_EcuInfo_t ecu_info_;
    std::vector<can_filter> filters_;

    std::mutex send_queue_mutex_;
    std::mutex recv_queue_mutex_;

    std::list<CanPacket> send_queue_;
    std::list<CanPacket> recv_queue_;
    std::condition_variable send_cond_;
    std::condition_variable recv_cond_;

    std::thread write_thread_;
    std::thread read_thread_;
    std::thread process_thread_;

    std::shared_ptr<PacketParser> m_parser;
};


} // end of diag
} // end of netaos
} // end of hozon

#endif  // DOCAN_SOCKET_H_

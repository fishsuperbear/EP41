/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CanSocket cpp completion
 */

#include "can_socket.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <linux/can/error.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "packet_parser.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    CanSocket::CanSocket(N_EcuInfo_t ecuInfo)
        : m_fd(-1)
        , stopFlag_(true)
        , ecu_info_(ecuInfo)
        , filters_()
        , send_queue_mutex_()
        , recv_queue_mutex_()
        , send_queue_()
        , recv_queue_()
        , write_thread_()
        , read_thread_()
        , process_thread_()
    {
        m_parser = std::make_shared<PacketParser>();
        SetupCanSocket(ecuInfo);
    }

    CanSocket::~CanSocket()
    {
        if (!IsStop()) {
            Stop();
        }

        if (m_fd > 0) {
            close(m_fd);
            m_fd = -1;
        }
        recv_queue_.clear();
        send_queue_.clear();
        m_parser = nullptr;
    }

    int32_t CanSocket::Init()
    {
        return 0;
    }

    int32_t CanSocket::Start()
    {
        DOCAN_LOG_I("ecu: %s, canif: %s, Start()~~", ecu_info_.ecu_name.c_str(), ecu_info_.if_name.c_str());
        stopFlag_ = false;
        write_thread_ = std::thread(&CanSocket::WriteThread, this);
        read_thread_ = std::thread(&CanSocket::ReadThread, this);
        // if (nullptr != m_parser) {
        //     process_thread_ = std::thread(&CanSocket::ProcessThread, this);
        // }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 0;
    }

    int32_t CanSocket::Stop()
    {
        DOCAN_LOG_I("ecu: %s, canif: %s, Stop()~~", ecu_info_.ecu_name.c_str(), ecu_info_.if_name.c_str());
        stopFlag_ = true;
        send_cond_.notify_all();
        recv_cond_.notify_all();
        if (write_thread_.joinable()) {
            write_thread_.join();
        }
        if (read_thread_.joinable()) {
            read_thread_.join();
        }
        // if (process_thread_.joinable()) {
        //     process_thread_.join();
        // }
        return 0;
    }

    bool CanSocket::IsStop()
    {
        return stopFlag_;
    }

    N_EcuInfo_t CanSocket::GetEcuInfo()
    {
        return ecu_info_;
    }

    bool CanSocket::GetIfFlags(const std::string& ifname)
    {
        int32_t canfd = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (canfd <= 0) {
            return false;
        }

        struct ifreq ifr;
        memset(&ifr, 0x00, sizeof(ifr));
        memcpy(ifr.ifr_name, ifname.c_str(), ifname.size());
        if (ioctl(canfd, SIOCGIFFLAGS, &ifr)) {
            return false;
        }

        return ((uint8_t)(ifr.ifr_flags & IFF_RUNNING) != 0x00);
    }

    int32_t CanSocket::SetIfFlags(const std::string& ifname, bool up)
    {
        int32_t ret = -1;
        uint16_t flag = up ? IFF_UP : ~IFF_UP;
        int32_t canfd = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (canfd <= 0) {
            return ret;
        }

        struct ifreq ifr;
        memset(&ifr, 0x00, sizeof(ifr));
        memcpy(ifr.ifr_name, ifname.c_str(), ifname.size());
        ret = ioctl(canfd, SIOCGIFFLAGS, &ifr);
        if (ret < 0) {
            return ret;
        }

        ifr.ifr_ifru.ifru_flags |= flag;
        ret = ioctl(canfd, SIOCSIFFLAGS, &ifr);
        return ret;
    }

    int32_t CanSocket::SetPacketParser(std::shared_ptr<PacketParser> parser)
    {
        m_parser = parser;
        return 0;
    }

    int32_t CanSocket::RecvCanFrame(CanPacket& packet)
    {
        // DIAG_LOG_INFO << "RecvCanFrame canfd: " << m_fd;
        int32_t ret = -1;
        if (-1 == m_fd) {
            return ret;
        }

        struct pollfd fds[1];
        fds[0].fd = m_fd;
        fds[0].events = POLLIN | POLLNVAL | POLLERR | POLLHUP;
        static const int32_t POLLIN_TIMEOUT = 500;  // (msec)
        ret = poll(fds, 1, POLLIN_TIMEOUT);
        if (ret < 0) {
            DOCAN_LOG_E("poll(fds, 1, POLL_TIMEOUT) error: %s", strerror(errno));
            return ret;
        }

        if (ret == 0) {
            // DIAG_LOG_DEBUG << "poll() timeout: " << POLLIN_TIMEOUT << " msec";
            return 0;
        }

        if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) ||
            (fds[0].revents & POLLNVAL)) {
            DOCAN_LOG_E("poll() reports POLLERR|POLLHUP|POLLNVAL error");
            return -1;
        }

        if (fds[0].revents & POLLIN) {
            struct timeval tstamp{ 0 };
            iovec iov = {
                .iov_base = static_cast<void *>(&packet.frame),
                .iov_len = sizeof(packet.frame)
            };

            const std::uint32_t controlSize = 512U;
            char controlBuf[CMSG_SPACE(controlSize)];
            msghdr canMsg = {
                .msg_name = nullptr,
                .msg_namelen = 0U,
                .msg_iov = &iov,
                .msg_iovlen = 1U,
                .msg_control = controlBuf,
                .msg_controllen = sizeof(controlBuf),
                .msg_flags = 0,
            };

            ret = static_cast<int32_t>(recvmsg(m_fd, &canMsg, 0));
            if (ret < 0) {
                DOCAN_LOG_E("recvmsg(m_fd, &canMsg, 0) can faild, canfd: %d, errno: %s", m_fd, strerror(errno));
                return ret;
            }
            struct cmsghdr *cmsg = CMSG_FIRSTHDR(&canMsg);
            if (cmsg != nullptr) {
                tstamp = *(reinterpret_cast<timeval*>(CMSG_DATA(cmsg)));
            }

            packet.sec = tstamp.tv_sec;
            packet.sec = tstamp.tv_usec * 1000;
            packet.len = ret;

            DOCAN_LOG_D("poll() read  ret: %d, ecu: %s, canif: %s, canid: %X, data: [%02X %02X %02X %02X %02X %02X %02X %02X]",
                ret, ecu_info_.ecu_name.c_str(), ecu_info_.if_name.c_str(), packet.frame.can_id, packet.frame.data[0], packet.frame.data[1], packet.frame.data[2],
                packet.frame.data[3], packet.frame.data[4], packet.frame.data[5], packet.frame.data[6], packet.frame.data[7]);

        }

        return ret;
    }

    int32_t CanSocket::AddSendPacket(const CanPacket& packet)
    {
        {
            std::unique_lock<std::mutex> lck(send_queue_mutex_);
            send_queue_.push_back(packet);
        }
        send_cond_.notify_all();
        DOCAN_LOG_D("Cansocket ecu: %s, canif: %s, canid: %x, AddSendQueue current queue size: %ld",
            ecu_info_.ecu_name.c_str(), ecu_info_.if_name.c_str(), packet.frame.can_id, send_queue_.size());
        return send_queue_.size();
    }

    int32_t CanSocket::AddSendQueue(const std::vector<CanPacket>& queue)
    {
        {
            std::unique_lock<std::mutex> lck(send_queue_mutex_);
            send_queue_.insert(send_queue_.end(), queue.begin(), queue.end());
        }
        send_cond_.notify_all();
        DOCAN_LOG_D("Cansocket ecu: %s, canif: %s, canid: %x, AddSendQueue current queue size: %ld",
            ecu_info_.ecu_name.c_str(), ecu_info_.if_name.c_str(), queue[0].frame.can_id, send_queue_.size());
        return send_queue_.size();
    }

    int32_t CanSocket::SendCanFrame(const CanPacket& packet)
    {
        // DOCAN_LOG_I("SendCanFrame frame data len: %d", packet.frame.can_dlc);
        int32_t ret = -1;
        if (-1 == m_fd) {
            return ret;
        }

        // write all the data to fd by most
        struct pollfd fds[1];
        fds[0].fd = m_fd;
        fds[0].events = POLLOUT | POLLNVAL | POLLERR | POLLHUP;
        static const int32_t POLLOUT_TIMEOUT = 10;  // one second (in msec)
        ret = poll(fds, 1, POLLOUT_TIMEOUT);
        if (ret < 0) {
            DOCAN_LOG_E("poll(fds, 1, POLL_TIMEOUT) error: %s", strerror(errno));
            return ret;
        }

        if (ret == 0) {
            // DIAG_LOG_DEBUG << "poll() timeout: " << POLLIN_TIMEOUT << " msec";
            return 0;
        }

        if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) ||
            (fds[0].revents & POLLNVAL)) {
            DOCAN_LOG_E("poll() reports POLLERR|POLLHUP|POLLNVAL error");
            return -1;
        }

        if (fds[0].revents & POLLOUT) {
            ret = write(m_fd, &packet.frame, sizeof(packet.frame));  // can frame
            if (ret < 0) {
                DOCAN_LOG_E("write can fd faild, canfd: %d, if_name: %s, errno: %s", m_fd, ecu_info_.if_name.c_str(), strerror(errno));
                return ret;
            }

            DOCAN_LOG_D("poll() write ret: %d, ecu: %s, canif: %s, canid: %X, data: [%02X %02X %02X %02X %02X %02X %02X %02X]",
                ret, ecu_info_.ecu_name.c_str(), ecu_info_.if_name.c_str(), packet.frame.can_id, packet.frame.data[0], packet.frame.data[1], packet.frame.data[2],
                packet.frame.data[3], packet.frame.data[4], packet.frame.data[5], packet.frame.data[6], packet.frame.data[7]);

        }

        return ret;
    }

    int32_t CanSocket::SetRecvTimeout(const struct timeval &tv) const
    {
        int32_t ret = -1;
        ret = setsockopt(m_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, static_cast<socklen_t>(sizeof(tv)));
        if (ret < 0) {
            DOCAN_LOG_E("setsockopt(m_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) faild, canfd: %d, errno: %s", m_fd, strerror(errno));
        }
        return ret;
    }

    int32_t CanSocket::SetCanFilters(const std::vector<can_filter> &filters)
    {
        int32_t ret = -1;
        if (filters.empty()) {
            return ret;
        }
        filters_ = filters;
        const auto itemSize = static_cast<socklen_t>(sizeof(can_filter));
        const auto filterSize = static_cast<socklen_t>(static_cast<socklen_t>(filters.size()) * itemSize);
        ret = setsockopt(m_fd, SOL_CAN_RAW, CAN_RAW_FILTER, filters.data(), filterSize);
        if (ret < 0) {
            DOCAN_LOG_E("setsockopt(m_fd, SOL_CAN_RAW, CAN_RAW_FILTER, filters.data(), filterSize) faild, canfd: %d, errno: %s", m_fd, strerror(errno));
        }
        return ret;
    }

    std::list<CanPacket>& CanSocket::GetRecvQueue()
    {
        DOCAN_LOG_I("GetRecvQueue, fd: %d, recv_queue_ size: %ld", m_fd, recv_queue_.size());
        std::unique_lock<std::mutex> lck(recv_queue_mutex_);
        return recv_queue_;
    }

    int32_t CanSocket::SetupCanSocket(const N_EcuInfo_t& ecuInfo)
    {
        // can socket device
        int32_t ret = -1;
        if ("" == ecuInfo.if_name) {
            DOCAN_LOG_E("Can if invalid | canid is invalid!!");
            return ret;
        }

        DOCAN_LOG_I("ecuInfo.ecu_name:     %s", ecuInfo.ecu_name.c_str());
        DOCAN_LOG_I("ecuInfo.if_name:      %s", ecuInfo.if_name.c_str());
        DOCAN_LOG_I("ecuInfo.logicalAddr:  %X", ecuInfo.address_logical);
        DOCAN_LOG_I("ecuInfo.functionAddr: %X", ecuInfo.address_functional);
        DOCAN_LOG_I("ecuInfo.canid_tx:     %X", ecuInfo.canid_tx);
        DOCAN_LOG_I("ecuInfo.canid_rx:     %X", ecuInfo.canid_rx);

        if (m_fd > 0 ) {
            DOCAN_LOG_W("Can socket is existed, fd : %d", m_fd);
            return m_fd;
        }

        int32_t canfd = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
        DOCAN_LOG_I("socket(PF_CAN, SOCK_RAW, CAN_RAW) canfd:  %d", canfd);
        if (canfd <= 0) {
            DOCAN_LOG_E("Create socket(PF_CAN, SOCK_RAW, CAN_RAW) failed, canfd: %d, errno: %s", canfd, strerror(errno));
            return canfd;
        }

        do {
            struct ifreq ifr;
            memset(&ifr, 0x00, sizeof(ifr));
            memcpy(ifr.ifr_name, ecuInfo.if_name.c_str(), ecuInfo.if_name.size());
            ret = ioctl(canfd, SIOCGIFINDEX, &ifr);
            if (ret < 0) {
                DOCAN_LOG_E("ioctl(fd, SIOCGIFINDEX, &ifr) failed, canfd: %d, errno: %s", canfd, strerror(errno));
                break;
            }

            const int32_t canfd_flag = 1;
            ret = setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &canfd_flag,
                            static_cast<socklen_t>(sizeof(canfd_flag)));
            if (ret < 0) {
                DOCAN_LOG_E("setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &1, 4) failed, canfd: %d, errno: %s", canfd, strerror(errno));
                break;
            }

            // can_err_mask_t err_mask = (CAN_ERR_TX_TIMEOUT | CAN_ERR_CRTL | CAN_ERR_PROT
            //     | CAN_ERR_TRX | CAN_ERR_ACK | CAN_ERR_BUSOFF | CAN_ERR_BUSERROR);
            // ret = setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_ERR_FILTER, &err_mask,
            //                 static_cast<socklen_t>(sizeof(err_mask)));
            // if (ret < 0) {
            //     DOCAN_LOG_E("setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_ERR_FILTER, &err_mask, sizeof(err_mask)) failed, canfd: %d, errno: %s", canfd, strerror(errno));
            //     break;
            // }

            struct sockaddr_can addr = {};
            addr.can_family = static_cast<__kernel_sa_family_t>(AF_CAN);
            addr.can_ifindex = ifr.ifr_ifindex;
            ret = bind(canfd, reinterpret_cast<sockaddr*>(&addr), static_cast<socklen_t>(sizeof(addr)));
            if (ret < 0) {
                DOCAN_LOG_E("bind(canfd, &addr, sizeof(addr)) failed, canfd: %d, errno: %s", canfd, strerror(errno));
                break;
            }

            const int32_t timestamp_flag = 1;
            ret = setsockopt(canfd, SOL_SOCKET, SO_TIMESTAMP, &timestamp_flag,
                            static_cast<socklen_t>(sizeof(timestamp_flag)));
            if (ret < 0) {
                DOCAN_LOG_E("setsockopt(canfd, SOL_SOCKET, SO_TIMESTAMP, &timestamp_flag, sizeof(timestamp_flag)) failed, canfd: %d, errno: %s", canfd, strerror(errno));
                break;
            }

            const int32_t enableFlag = 0;
            ret = setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &enableFlag,
                            static_cast<socklen_t>(sizeof(enableFlag)));
            if (ret < 0) {
                DOCAN_LOG_E("setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &enableFlag, sizeof(enableFlag)) failed, canfd: %d, errno: %s", canfd, strerror(errno));
                break;
            }

            if (filters_.size() > 0) {
                const auto itemSize = static_cast<socklen_t>(sizeof(can_filter));
                const auto filterSize = static_cast<socklen_t>(static_cast<socklen_t>(filters_.size()) * itemSize);
                ret = setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_FILTER, filters_.data(), filterSize);
                if (ret < 0) {
                    DOCAN_LOG_E("setsockopt(m_fd, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, filterSize) faild, canfd: %d, errno: %s", canfd, strerror(errno));
                    break;
                }
            }
            else {
                can_filter filter;
                filter.can_id = ecu_info_.canid_rx;
                filter.can_mask = 0x700;
                ret = setsockopt(canfd, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, static_cast<socklen_t>(sizeof(can_filter)));
                if (ret < 0) {
                    DOCAN_LOG_E("setsockopt(m_fd, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, filterSize) faild, canfd: %d, errno: %s", canfd, strerror(errno));
                    break;
                }
            }

        } while (0);

        if (ret < 0) {
            close(canfd);
            canfd = -1;
            return canfd;
        }

        DOCAN_LOG_I("SetupCanSocket successful canfd: %d", canfd);
        m_fd = canfd;
        return canfd;
    }

    int32_t CanSocket::ReadThread() {
        DOCAN_LOG_I("ReadThread successful canfd: %d", m_fd);
        int32_t ret = -1;
        if (m_fd < 0) {
            DOCAN_LOG_E("ReadThread invalid canfd: %d", m_fd);
            return ret;
        }
        CanPacket packet;
        while (!stopFlag_) {
            memset(&packet, 0x00, sizeof(packet));
            if (RecvCanFrame(packet) <= 0) {
                continue;
            }

            ret = packet.frame.can_dlc;
            if (nullptr != m_parser) {
                m_parser->ParserCanPacket(ecu_info_, packet);
            }
            else {
                std::unique_lock<std::mutex> lck(recv_queue_mutex_);
                if (recv_queue_.size() < DIAG_QUQUE_BUFF_SIZE) {
                    recv_queue_.push_back(packet);
                }
            }
        }
        return ret;
    }

    int32_t CanSocket::WriteThread() {
        DOCAN_LOG_I("WriteThread successful canfd: %d", m_fd);
        int32_t ret = -1;
        if (m_fd < 0) {
            DOCAN_LOG_E("WriteThread invalid canfd: %d", m_fd);
            return ret;
        }

        while (!stopFlag_) {
            std::unique_lock<std::mutex> lck(send_queue_mutex_);
            send_cond_.wait_for(lck, std::chrono::milliseconds(100));
            while (send_queue_.size() > 0 && !stopFlag_) {
                CanPacket packet = send_queue_.front();
                send_queue_.pop_front();
                DOCAN_LOG_D("canid: %x, STmin: %d, left queue size: %ld.", packet.frame.can_id, packet.STmin, send_queue_.size());

                packet.frame.can_id = ecu_info_.canid_tx;
                packet.frame.__pad = 0xAA;
                packet.frame.can_dlc = 0x08;
                uint32_t retry = 0;
                do {
                    int32_t ret = SendCanFrame(packet);
                    if (ret < 0) {
                        // write failed
                        break;
                    }
                    else if (ret > 0) {
                        // write successful
                        if (packet.STmin >= 1 && packet.STmin <= 127) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(packet.STmin));
                        }
                        break;
                    }
                    else {
                        // current can not write need just little wait
                        if (retry >= 100) {
                            DOCAN_LOG_E("wait retry %d times poll each 10ms still failed.", retry);
                            break;
                        }
                        ++retry;
                        continue;
                    }

                } while (0);
            }
        }
        return ret;
    }


    int32_t CanSocket::ProcessThread()
    {
        DOCAN_LOG_I("packet process thread start");
        int32_t ret = -1;
        if (m_fd < 0) {
            DOCAN_LOG_E("ProcessThread invalid canfd: %d", m_fd);
            return ret;
        }
        if (nullptr == m_parser) {
            return ret;
        }
        CanPacket packet;
        while (!stopFlag_) {
            {
                std::unique_lock<std::mutex> lck(recv_queue_mutex_);
                if (recv_queue_.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                packet = recv_queue_.front();
                ret = m_parser->ParserCanPacket(ecu_info_, packet);
                recv_queue_.pop_front();
            }
        }
        return ret;
    }

} // end of diag
} // end of netaos
} // end of hozon

/* EOF */
#include "udp_base.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <sys/fcntl.h>

#include "common/logger.h"
// #include "faultmessage/lidar_fault_report.h"

namespace hozon {
namespace ethstack {
namespace lidar {

    UdpBase::UdpBase(EthernetSocketInfo socket)
        : m_fd(-1)
        , stopFlag_(false)
        , send_queue_mutex_()
        , recv_queue_mutex_()
        , send_queue_()
        , recv_queue_()
        , poll_thread_()
        , process_thread_()
        , m_dispatcher (nullptr)
    {
        m_socketInfo.frame_id = socket.frame_id;
        m_socketInfo.if_name = socket.if_name;
        m_socketInfo.local_ip = socket.local_ip;
        m_socketInfo.local_port = socket.local_port;
        m_socketInfo.remote_ip = socket.remote_ip;
        m_socketInfo.remote_port = socket.remote_port;
        m_socketInfo.multicast = socket.multicast;
        m_dispatcher = std::shared_ptr<UdpDispatcher>(new UdpDispatcher());
        if (SetupEthernetSocket(socket) < 0) {
            LIDAR_LOG_ERROR << "SetupEthernetSocket failed";
            LidarFaultReport::Instance().ReportAbstractFault(AbstractFaultObject::SOCKET_ERROR, CURRENT_FAULT);

            uint8_t lidar_history_status = hozon::ethstack::lidar::LidarStatusReport::Instance().GetLidarStatus();
            uint8_t lidar_current_status = 2;
            if (lidar_history_status != lidar_current_status){
                hozon::netaos::cfg::ConfigParam::Instance()->SetParam("system/lidar_status", lidar_current_status);
                LIDAR_LOG_TRACE << "success write lidar work status to cfg.";
                hozon::ethstack::lidar::LidarStatusReport::Instance().SetLidarStatus(lidar_current_status);
                LIDAR_LOG_TRACE << "success write lidar work status to LidarStatus class.";
            }
        }
    }

    UdpBase::~UdpBase()
    {
        if (m_fd > 0) {
            close(m_fd);
            m_fd = -1;
        }
    }

    int32_t UdpBase::Init()
    {
        return 0;
    }

    int32_t UdpBase::Start()
    {
        LIDAR_LOG_INFO << "UdpBase PollThread Create. socket_port : " << m_socketInfo.local_port;
        stopFlag_ = false;
        poll_thread_ = std::thread(&UdpBase::PollThread, this);
        pthread_setname_np(poll_thread_.native_handle(), (std::string("PollThread")).c_str());
        return 0;
    }

    int32_t UdpBase::StartProcess()
    {
        LIDAR_LOG_INFO << "UdpBase process_thread Create. socket_port : " << m_socketInfo.local_port;
        stopFlag_ = false;
        process_thread_ = std::thread(&UdpBase::ProcessThread, this);
        pthread_setname_np(process_thread_.native_handle(), (std::string("ProcessThread")).c_str());
        return 0;
    }


    int32_t UdpBase::Stop()
    {
        stopFlag_ = true;
        process_condition_.notify_all();
        if (poll_thread_.joinable()) {
            poll_thread_.join();
            LIDAR_LOG_INFO << "poll_thread_ Stopped()";
        }
        if (process_thread_.joinable()) {
            process_thread_.join();
            LIDAR_LOG_INFO << "process_thread_ Stopped()";
        }
        LIDAR_LOG_INFO << "UdpBase PollThread Stop. socket_port : " << m_socketInfo.local_port;
        return 0;
    }


    bool UdpBase::IsStop() const
    {
        return stopFlag_;
    }

    int32_t UdpBase::GetSocketFd() const
    {
        return m_fd;
    }

    int32_t UdpBase::RecvEthernetPacket(EthernetPacket* packet)
    {
        int32_t ret = -1;
        // if (-1 == m_fd || nullptr == packet) {
        //     LIDAR_LOG_INFO << " m_fd is: "<< m_fd;
        //     LIDAR_LOG_ERROR << "-1 == m_fd || nullptr == packet";
        //     return ret;
        // }
        struct pollfd fds[1];               //程序只对轮询一个文件描述符的事件感兴趣
        fds[0].fd = m_fd;        
        fds[0].events = POLLIN | POLLNVAL | POLLERR | POLLHUP;      //“POLLIN”（可供读取的数据）、“POLLNVAL”（无效请求，例如未打开的文件描述符）、“POLLERR”（错误条件）和“POLLHUP”（挂起条件）。程序对这些事件感兴趣，以相应地处理套接字。
        static const int POLLIN_TIMEOUT = 1000;  // one second (in msec)
        ret = poll(fds, 1, POLLIN_TIMEOUT);      //“poll”函数调用。它将等待文件描述符“m_fd”上发生的事件，并在“POLLIN_TIMEOUT”指定的超时后返回。
        if (ret < 0) {
            if (errno != EINTR) {
                LIDAR_LOG_WARN << "poll(fds, 1, POLL_TIMEOUT) error:" << strerror(errno);
            }
            return ret;
        }

        if (ret == 0) {
            LIDAR_LOG_INFO << "poll() timeout: " << POLLIN_TIMEOUT << " msec";
            return 0;
        }

        if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) ||
            (fds[0].revents & POLLNVAL)) {
            LIDAR_LOG_WARN << "poll() reports POLLERR|POLLHUP|POLLNVAL error";
            return -1;
        }
        if (fds[0].revents & POLLIN) {
            memset(packet->data, 0x00, sizeof(packet->data));
            struct sockaddr_in src_addr;
            socklen_t len = sizeof(src_addr);
            memset(&src_addr, 0x00, sizeof(src_addr));
            ret = recvfrom(m_fd, packet->data, sizeof(packet->data), 0, reinterpret_cast<struct sockaddr*>(&src_addr), static_cast<socklen_t*>(&len));  // if support can TP.
            if (ret <= 0) {
                if (EWOULDBLOCK != errno) {
                    LIDAR_LOG_ERROR << "recv(m_fd, &packet.data, sizeof(packet.data), MSG_DONTWAIT) faild, m_fd: " << m_fd << "if_name: " << m_socketInfo.if_name << ", errno: " << strerror(errno);
                }
                return ret;
            }
            packet->len = ret;
            
        }
        // fill into the data info
        struct timespec tp;
        clock_gettime(CLOCK_REALTIME, &tp);
        packet->nsec                 = tp.tv_nsec;
        packet->sec                  = tp.tv_sec;
        return ret;
    }

    int32_t UdpBase::SendEthernetPacket(const EthernetPacket* packet)

    {

        int32_t ret = -1;
        if (-1 == m_fd || nullptr == packet) {
            LIDAR_LOG_ERROR << "SendEthernetPacket -1 == m_fd || nullptr == packet";
            return ret;
        }
        LIDAR_LOG_DEBUG << "SendEthernetPacket ethfd: " << m_fd << ", packet size: " << packet->len;

        int32_t left = packet->len;
        // write all the data to fd by most
        while (left > 0) {
            struct pollfd fds[1];
            fds[0].fd = m_fd;
            fds[0].events = POLLOUT | POLLNVAL | POLLERR | POLLHUP;
            static const int POLLOUT_TIMEOUT = 10;  // one second (in msec)
            ret = poll(fds, 1, POLLOUT_TIMEOUT);
            if (ret < 0) {
                if (errno != EINTR) {
                    LIDAR_LOG_WARN << "poll(fds, 1, POLL_TIMEOUT) error:" << strerror(errno);
                }
                return ret;
            }

            if (ret == 0) {
                // LIDAR_LOG_DEBUG << "poll() timeout: " << POLLIN_TIMEOUT << " msec";
                return 0;
            }

            if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) ||
                (fds[0].revents & POLLNVAL)) {
                LIDAR_LOG_WARN << "poll() reports POLLERR|POLLHUP|POLLNVAL error:" << strerror(errno);
                return -1;
            }

            if (fds[0].revents & POLLOUT) {
                struct sockaddr_in server_addr;
                memset(&server_addr, 0x00, sizeof(server_addr));
                server_addr.sin_family = AF_INET;
                server_addr.sin_port = htons(m_socketInfo.remote_port);
                server_addr.sin_addr.s_addr = inet_addr(m_socketInfo.remote_ip.c_str());
                ret = sendto(m_fd, &packet->data[packet->len - left], left, 0, (struct sockaddr*)&server_addr, sizeof(server_addr));  // if support can TP.
                if (ret < 0) {
                    LIDAR_LOG_ERROR << "write can fd faild, len: " << packet->len << " m_fd: " << m_fd << ", errno: " << strerror(errno);
                    return ret;
                }
                left -= ret;
            }
        }

        return packet->len;
    }
    
    // 组播方式
    int32_t UdpBase::SetupEthernetSocket(const EthernetSocketInfo& socket)
    {
        // can socket device
        int32_t ret = -1;
        if ("" == socket.if_name || 0x0000 == socket.local_port) {
            LIDAR_LOG_ERROR << "Ethernet if_name|local_port is invalid !!";
            return ret;
        }

        if (m_fd > 0) {
            LIDAR_LOG_ERROR << "socket is existed, fd : " << m_fd;
            return m_fd;
        }

        LIDAR_LOG_INFO << "socket.frame_id:    " << socket.frame_id;
        LIDAR_LOG_INFO << "socket.if_name:     " << socket.if_name;
        LIDAR_LOG_INFO << "socket.local_ip:    " << socket.local_ip;
        LIDAR_LOG_INFO << "socket.local_port:  " << socket.local_port;
        LIDAR_LOG_INFO << "socket.remote_ip:   " << socket.remote_ip;
        LIDAR_LOG_INFO << "socket.remote_port: " << socket.remote_port;
        LIDAR_LOG_INFO << "socket.multecast:   " << socket.multicast;
        
        int32_t ethfd = ::socket(AF_INET, SOCK_DGRAM, 0);                       //创建socket
        if (ethfd <= 0) {
            LIDAR_LOG_ERROR << "socket is existed, if_name: " << socket.if_name << ", errno: " << strerror(errno);
            return 0;
        }

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(socket.local_port);  // 组播端口号
        addr.sin_addr.s_addr = inet_addr(socket.multicast.c_str());  // 组播地址

        // inet_pton(AF_INET, "224.224.224.244", &addr.sin_addr);  // 组播地址

        if (bind(ethfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            LIDAR_LOG_WARN << "bind(ethfd, reinterpret_cast<sockaddr*>(&sa), sizeof(sockaddr)) failed, ethfd: " << ethfd << ", errno: " << strerror(errno);
            close(ethfd);
            return 0;
        }

        //设置缓存大小
        int nRecvBuf = 400000000;
        setsockopt(ethfd, SOL_SOCKET, SO_RCVBUF, (const char *)&nRecvBuf,sizeof(int));

        struct ip_mreq mreq;
        memset(&mreq, 0, sizeof(mreq));
        mreq.imr_multiaddr.s_addr = inet_addr(socket.multicast.c_str());  // 组播地址
        mreq.imr_interface.s_addr = inet_addr(socket.local_ip.c_str());  // 本地接口地址

        if (setsockopt(ethfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
            LIDAR_LOG_WARN << "setsockopt(ethfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (const char*)&multicast_addr, sizeof(multicast_addr)) failed, ethfd: " << ethfd << ", errno: " << strerror(errno);
            close(ethfd);
            return 0;
        }

        m_fd = ethfd;
        return ethfd;
    }



    std::list<std::shared_ptr<EthernetPacket>>& UdpBase::GetRecvQueue()
    {
        LIDAR_LOG_DEBUG << "GetRecvQueue, fd : " << m_fd << ", recv_queue_ size: " << recv_queue_.size();
        std::unique_lock<std::mutex> lck(recv_queue_mutex_);
        return recv_queue_;
    }


    int32_t UdpBase::PollThread()
    {
        int32_t ret = -1;

        while (!stopFlag_) {
            std::shared_ptr<EthernetPacket> packet = std::make_shared<EthernetPacket>();

            if (nullptr == packet) {
                LIDAR_LOG_ERROR << "Lidar EthernetPacket Create failed.";
                LidarFaultReport::Instance().ReportAbstractFault(AbstractFaultObject::DATA_HANDLE_ACQUISITION_FAILED, CURRENT_FAULT);

                uint8_t lidar_history_status = hozon::ethstack::lidar::LidarStatusReport::Instance().GetLidarStatus();
                uint8_t lidar_current_status = 2;
                if (lidar_history_status != lidar_current_status){
                    hozon::netaos::cfg::ConfigParam::Instance()->SetParam("system/lidar_status", lidar_current_status);
                    LIDAR_LOG_TRACE << "success write lidar work status to cfg.";
                    hozon::ethstack::lidar::LidarStatusReport::Instance().SetLidarStatus(lidar_current_status);
                    LIDAR_LOG_TRACE << "success write lidar work status to LidarStatus class.";
                }
                return ret;
            }

            if (RecvEthernetPacket(packet.get()) <= 0) {
                // if (m_socketInfo.local_port == LOCAL_POINT_CLOUD_PORT) {
                    LIDAR_LOG_ERROR << " Lidar period status message recv timeout.";
                    LidarFaultReport::Instance().ReportCommunicationFault(CommunicationFaultObject::COMM_LOST, CURRENT_FAULT);

                    uint8_t lidar_history_status = hozon::ethstack::lidar::LidarStatusReport::Instance().GetLidarStatus();
                    uint8_t lidar_current_status = 2;
                    if (lidar_history_status != lidar_current_status){
                        hozon::netaos::cfg::ConfigParam::Instance()->SetParam("system/lidar_status", lidar_current_status);
                        LIDAR_LOG_TRACE << "success write lidar work status to cfg.";
                        hozon::ethstack::lidar::LidarStatusReport::Instance().SetLidarStatus(lidar_current_status);
                        LIDAR_LOG_TRACE << "success write lidar work status to LidarStatus class.";
                    }
                // }
                continue;
            }

            std::unique_lock<std::mutex> lck(recv_queue_mutex_);
            recv_queue_.push_back(packet);
            process_condition_.notify_all();
            

            // if (nullptr != m_dispatcher) {
            //     m_dispatcher->Parse(m_socketInfo.local_port, packet);
            // }

        }
        LIDAR_LOG_INFO << "PollThread exit. fd: " << m_fd;
        return ret;
    }

    int32_t UdpBase::ProcessThread()
    {
        LIDAR_LOG_INFO << "packet process thread start";
        int32_t ret = -1;
        if (nullptr == m_dispatcher) {
            return ret;
        }
        while (!stopFlag_) {
            std::shared_ptr<EthernetPacket> packet = nullptr;
            {
                std::unique_lock<std::mutex> lck(recv_queue_mutex_);
                if (recv_queue_.empty()) {
                    process_condition_.wait(lck);
                    continue;
                }
                packet= recv_queue_.front();
                recv_queue_.pop_front();
            }
            ret = m_dispatcher->Parse(m_socketInfo.local_port, packet);
        }
        LIDAR_LOG_INFO << "packet process thread stop";
        return ret;
    }

}
}
}

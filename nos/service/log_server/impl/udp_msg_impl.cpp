#include "log_server/impl/udp_msg_impl.h"
#include "log_server/log/log_server_logger.h"

#include <arpa/inet.h>
#include <unistd.h>

namespace hozon {
namespace netaos {
namespace logserver {

UdpMsgImpl::UdpMsgImpl()
{
}

int32_t
UdpMsgImpl::Init(const std::string& ip, const uint32_t& port)
{
    LOG_SERVER_INFO << "UdpMsgImpl::Init";
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        LOG_SERVER_ERROR << "Failed to create UDP socket.";
        return -1;
    }
    // 设置超时5s
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == -1) {
        LOG_SERVER_ERROR << "Error setting socket timeout: " << strerror(errno);
        return -2;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(ip.c_str());
    server_addr.sin_port = htons(port);

    if (bind(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        LOG_SERVER_ERROR << "Failed to bind UDP socket";
        return -3;
    }
    is_connected_ = true;
    return 0;
}

int32_t
UdpMsgImpl::DeInit()
{
    LOG_SERVER_INFO << "UdpMsgImpl::DeInit";
    LOG_SERVER_INFO << "UdpMsgImpl::DeInit Done";
    return 0;
}

void
UdpMsgImpl::SetMcuLogCallback(McuLogCallback callback)
{
    mcu_log_callback_ = std::move(callback);
}

int32_t
UdpMsgImpl::Start() {
    LOG_SERVER_INFO << "UdpMsgImpl::Start";
    waitReq_ = std::thread(&UdpMsgImpl::WaitRequest, this);
    LOG_SERVER_INFO << "UdpMsgImpl::Start Done";
    return 0;
}

int32_t
UdpMsgImpl::WaitRequest()
{
    if (!is_connected_) {
        LOG_SERVER_ERROR << "Server is not ready running.";
        return -1;
    }
    std::array<uint8_t, sizeof(McuLog)> buffer{};
    while (!is_quit_) {
        ssize_t bytes_received = recvfrom(socket_fd_, buffer.data(), buffer.size(), 0, nullptr, nullptr);
        if (bytes_received < 0) {
            LOG_SERVER_TRACE << "Receive UDP data Time Out.";
            continue;
        }
        std::vector<uint8_t> udpDate{buffer.begin(), buffer.end()};
        McuLog mcuLog{};
        if (ParseLog(udpDate, mcuLog)) {
            if (mcu_log_callback_) {
                mcu_log_callback_(mcuLog);
            }
        } else {
            LOG_SERVER_ERROR << "Failed to ParseLog.";
            continue;
        }
    }
    LOG_SERVER_DEBUG << "UdpMsgImpl::WaitRequest() end";
    return 0;
}


int32_t
UdpMsgImpl::Stop()
{
    LOG_SERVER_INFO << "UdpMsgImpl::Stop() begin";
    is_quit_ = true;

    if (is_connected_) {
        shutdown(socket_fd_, SHUT_RDWR);
    }

    if (waitReq_.joinable()) {
        waitReq_.join();
    }
    LOG_SERVER_DEBUG << "UdpMsgImpl::Stop() thread quit";
    if (socket_fd_ != -1) {
        close(socket_fd_);
        socket_fd_ = -1;
    }

    is_connected_ = false;
    LOG_SERVER_INFO << "UdpMsgImpl::Stop() end";

    return 0;
}


bool
UdpMsgImpl::ParseLog(const std::vector<uint8_t>& udpData, McuLog& log)
{
    if (udpData.size() != sizeof(McuLog)) {
        LOG_SERVER_ERROR << "udpData size is no correct.";
        return false;
    }
    std::memcpy(&log, udpData.data(), sizeof(McuLog));

    LOG_SERVER_TRACE << "appID :" << static_cast<int>(log.header.app_id);
    LOG_SERVER_TRACE << "ctxID :" << static_cast<int>(log.header.ctx_id);
    LOG_SERVER_TRACE << "length :" << log.header.length;
    LOG_SERVER_TRACE << "level :" << static_cast<int>(log.header.level);
    LOG_SERVER_TRACE << "seq :" << static_cast<int>(log.header.seq);
    LOG_SERVER_TRACE << "second :" << log.header.stamp.sec;
    LOG_SERVER_TRACE << "n second :" << log.header.stamp.nsec;
    std::string data{log.log.data(), log.log.data() + log.header.length};
    LOG_SERVER_TRACE << "log data :" << data;

    if (log.header.app_id > 0x02 || log.header.app_id < 0x01)
    {
        LOG_SERVER_ERROR << "appId is no correct.";
        return false;
    }

    if (log.header.ctx_id > 0x03)
    {
        LOG_SERVER_ERROR << "ctxId is no correct.";
        return false;
    }

    if (log.header.level > 0x06)
    {
        LOG_SERVER_ERROR << "level is no correct.";
        return false;
    }

    if (log.header.length > 300)
    {
        LOG_SERVER_ERROR << "length is no correct.";
        return false;
    }

    return true;
}


}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

#include "update_manager/mcu/mcu_uds_peri.h"
#include "update_manager/log/update_manager_logger.h"

#include <arpa/inet.h>
#include <unistd.h>

namespace hozon {
namespace netaos {
namespace update {
McuUdsPeri* McuUdsPeri::m_pInstance = nullptr;
std::mutex McuUdsPeri::m_mtx;

McuUdsPeri::McuUdsPeri()
{
}

McuUdsPeri::~McuUdsPeri()
{
}

McuUdsPeri*
McuUdsPeri::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new McuUdsPeri();
        }
    }

    return m_pInstance;
}

int32_t
McuUdsPeri::Init(const std::string& ip, const uint32_t& port)
{
    UM_INFO << "McuUdsPeri::Init.";
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        UM_ERROR << "Failed to create UDP socket.";
        return -1;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(ip.c_str());
    server_addr.sin_port = htons(port);

    if (bind(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        UM_ERROR << "Failed to bind UDP socket";
        return -3;
    }
    is_connected_ = true;
    UM_INFO << "McuUdsPeri::Init Done.";
    return 0;
}

int32_t
McuUdsPeri::DeInit()
{
    UM_INFO << "McuUdsPeri::DeInit.";
    mcu_uds_callback_ = nullptr;
    UM_INFO << "McuUdsPeri::DeInit Done.";
    return 0;
}

void
McuUdsPeri::SetMcuUdsCallback(McuUdsCallback callback)
{
    mcu_uds_callback_ = std::move(callback);
}

int32_t
McuUdsPeri::Start() {
    UM_INFO << "McuUdsPeri::Start.";
    waitReq_ = std::thread(&McuUdsPeri::WaitRequest, this);
    UM_INFO << "McuUdsPeri::Start Done.";
    return 0;
}

int32_t
McuUdsPeri::WaitRequest()
{
    if (!is_connected_) {
        UM_ERROR << "Server is not ready running.";
        return -1;
    }
    std::array<uint8_t, sizeof(McuUdsMsg)> buffer{};
    while (!is_quit_) {
        ssize_t bytes_received = recvfrom(socket_fd_, buffer.data(), buffer.size(), 0, nullptr, nullptr);
        if (bytes_received < 0) {
            UM_DEBUG << "Receive UDP data error.";
            continue;
        }
        std::vector<uint8_t> udpDate{buffer.begin(), buffer.end()};
        McuUdsMsg mcuUdsMsg{};
        if (ParseMsg(udpDate, mcuUdsMsg)) {
            if (mcu_uds_callback_) {
                mcu_uds_callback_(mcuUdsMsg);
            }
        } else {
            UM_ERROR << "Failed to ParseMsg.";
            continue;
        }
    }
    UM_DEBUG << "McuUdsPeri::WaitRequest() end";
    return 0;
}


int32_t
McuUdsPeri::Stop()
{
    UM_INFO << "McuUdsPeri::Stop.";
    is_quit_ = true;

    if (is_connected_) {
        struct sockaddr_in dest_addr;
        // 设置目标地址和端口号
        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_addr.s_addr = inet_addr(mcu_uds_ip.c_str()); // 目标IP地址
        dest_addr.sin_port = htons(mcu_uds_port); // 目标端口号
        // 发送消息
        UM_DEBUG << "McuUdsPeri::Stop() send quit udp msg begin";
        sendto(socket_fd_, "quitReceiveUds", std::string("quitReceiveUds").size(), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        UM_DEBUG << "McuUdsPeri::Stop() send quit udp msg end";
    }

    if (waitReq_.joinable()) {
        waitReq_.join();
    }
    UM_DEBUG << "McuUdsPeri::Stop() thread quit";
    if (socket_fd_ != -1) {
        close(socket_fd_);
        socket_fd_ = -1;
    }

    is_connected_ = false;
    UM_INFO << "McuUdsPeri::Stop Done.";
    return 0;
}


bool
McuUdsPeri::ParseMsg(const std::vector<uint8_t>& udpData, McuUdsMsg& msg)
{
    if (udpData.size() != sizeof(McuUdsMsg)) {
        UM_ERROR << "udpData size is no correct.";
        return false;
    }
    std::memcpy(&msg, udpData.data(), sizeof(McuUdsMsg));

    UM_DEBUG << "Mcu Uds msg is :" << UM_UINT8_VEC_TO_HEX_STRING(udpData);

    if (msg.at(0) != 0x28 || msg.at(2) != 0x01)
    {
        UM_ERROR << "[Sid] is no correct. receive sid is : " << static_cast<uint16_t>(msg.at(0));
        return false;
    }

    if (msg.at(1) != 0x00 && msg.at(1) != 0x03)
    {
        UM_ERROR << "[SubId] is no correct. receive subid is : " << static_cast<uint16_t>(msg.at(1));
        return false;
    }

    return true;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon

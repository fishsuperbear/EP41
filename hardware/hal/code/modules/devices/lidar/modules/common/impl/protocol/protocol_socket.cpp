#include "lidar/modules/common/impl/protocol/protocol_socket.h"

SocketProtocol::SocketProtocol()
{
}

SocketProtocol::~SocketProtocol()
{
    if (sockfd_ != -1)
    {
        close(sockfd_);
    }
}

bool SocketProtocol::init(const LidarConfig &config)
{
    if (sockfd_ != -1)
    {
        close(sockfd_);
    }

    config_ = config;

    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ == -1)
    {
        HW_LIDAR_LOG_ERR("lidar%d open socket failed! port: %d\n", config_.index, config_.port);
        return false;
    }

    int defRcvBufSize = -1;
    socklen_t optlen = sizeof(defRcvBufSize);
    if (getsockopt(sockfd_, SOL_SOCKET, SO_RCVBUF, &defRcvBufSize, &optlen) < 0)
    {
        HW_LIDAR_LOG_ERR("lidar%d getsockopt error! port: %d!\n", config_.index, config_.port);
        return false;
    }

    int rcvBufSize = 1024 * 1024 * 500;
    optlen = sizeof(rcvBufSize);
    if (setsockopt(sockfd_, SOL_SOCKET, SO_RCVBUF, &rcvBufSize, optlen) < 0)
    {
        HW_LIDAR_LOG_ERR("lidar%d setsockopt error! port: %d!\n", config_.index, config_.port);
        return false;
    }

    sockaddr_in my_addr;
    memset(&my_addr, 0, sizeof(my_addr));
    my_addr.sin_family = AF_INET;
    my_addr.sin_port = htons(uint16_t(config_.port));
    my_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd_, reinterpret_cast<sockaddr *>(&my_addr), sizeof(sockaddr)) == -1)
    {
        HW_LIDAR_LOG_ERR("lidar%d socket bind failed! port: %d!\n", config_.index, config_.port);
        return false;
    }

    // if (fcntl(sockfd_, F_SETFL, O_NONBLOCK | FASYNC) < 0)
    // {
    //     HW_LIDAR_LOG_ERR("lidar%d set non-block failed! port: %d!\n", config_.index, config_.port);
    //     return false;
    // }

    HW_LIDAR_LOG_ERR("lidar%d init socket success, port: %d!\n", config_.index, config_.port);
    return true;
}

int SocketProtocol::getLidarOriginDataRobosenseM1(Scan::Packet &packets)
{
    while (true)
    {
        ssize_t len = recvfrom(sockfd_, &(packets.data[0]), PACKET_DATA_LENGTH_ROBOSENSE_M1, 0, NULL, NULL);
        if (len < 0)
        {
            HW_LIDAR_LOG_ERR("lidar%d socket recvfrom failed! port: %d!\n", config_.index, config_.port);
            return RECEIVE_FAILED;
        }

        if (len == PACKET_DATA_LENGTH_ROBOSENSE_M1)
        {
            break;
        }
    }

    return RECEIVE_SUCCESS;
}

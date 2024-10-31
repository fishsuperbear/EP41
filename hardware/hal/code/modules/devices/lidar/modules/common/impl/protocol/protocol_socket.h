#ifndef HW_LIDAR_PROTOCOL_SOCKET_H
#define HW_LIDAR_PROTOCOL_SOCKET_H

#include <arpa/inet.h>
#include <errno.h>
#include <poll.h>
#include <unistd.h> // close
#include <string.h> // memset
#include <sys/socket.h>

#include "lidar/modules/common/hw_lidar_log_impl.h"
#include "lidar/modules/common/impl/utils/lidar_types.h"

#define RECEIVE_SUCCESS 0
#define RECEIVE_FAILED -1

#define PACKET_DATA_LENGTH_ROBOSENSE_M1 1210

class SocketProtocol
{
public:
    SocketProtocol();
    ~SocketProtocol();

    bool init(const LidarConfig &config);
    
    int getLidarOriginDataRobosenseM1(Scan::Packet &packets);

private:
    LidarConfig config_;
    int sockfd_;
};

#endif // HW_LIDAR_PROTOCOL_SOCKET_H

#include <errno.h>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/poll.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <linux/can/error.h>
#include <linux/can/raw.h>
#include <linux/can.h>
#include <net/if.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <thread>

#define   DIAG_MDC_SECURITY_ACCESS_APP_MASK          (0x23AEBEFD)
#define   DIAG_MDC_SECURITY_ACCESS_BOOT_MASK         (0xAB854A17)
#define   DIAG_CONTI_LRR_SECURITY_ACCESS_APP_MASK    (0x7878824A)
#define   DIAG_CONTI_LRR_SECURITY_ACCESS_BOOT_MASK   (0x7777834B)

// #define   DIAG_CHUHANG_LRR_SECURITY_ACCESS_APP_MASK  (0x9AB07B6A)
#define   DIAG_CHUHANG_LRR_SECURITY_ACCESS_APP_MASK  (0x7878824A)
#define   DIAG_CHUHANG_LRR_SECURITY_ACCESS_BOOT_MASK (0x526A3583)

#define LOG(format, ...) \
 { \
    char print_msg[1024]= { 0 };    \
    struct timeval tv;              \
    gettimeofday(&tv, nullptr);     \
    struct tm *timeinfo = localtime(&tv.tv_sec);        \
    uint32_t milliseconds = tv.tv_usec / 1000;          \
    char time_buf[64] = { 0 };                          \
    memset(time_buf, 0x00, sizeof(time_buf));           \
    memset(print_msg, 0x00, sizeof(print_msg));         \
    snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d ",        \
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,             \
        timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, milliseconds);          \
    snprintf(print_msg, sizeof(print_msg), (format), ##__VA_ARGS__);                   \
    printf("[%s] [%d %ld %s@%s(%d) | %s]\n", time_buf, getpid(), syscall(__NR_gettid), \
        __FUNCTION__, (nullptr == strrchr(__FILE__, '/')) ? __FILE__: (strrchr(__FILE__, '/') + 1), __LINE__, (print_msg)); \
 }


 ///  data struct defination
typedef struct EthernetSocketInfo {
    std::string frame_id;
    std::string if_name;
    std::string local_ip;
    std::string remote_ip;
    std::string multicast;
    uint16_t local_port;
    uint16_t remote_port;

} EthernetSocketInfo;

typedef struct EthernetPacket {
    uint32_t sec;
    uint32_t nsec;
    uint32_t len;
    uint8_t data[3690] = { 0 };
} EthernetPacket;

std::string ArrayToString(const uint8_t* data, uint32_t size)
{
    if (data == nullptr || size == 0) {
        return std::string("");
    }
    uint32_t len = size > 64 ? 64 : size;
    char* buf = new char[len*3 + 3];
    memset(buf, 0x00, len*3 + 3);
    for (uint32_t index = 0; index < len; ++index) {
        snprintf(buf + index*3, len*3 - index*3 , "%02X ", data[index]);
    }
    std::string str;
    if (size > 64) {
        buf[len*3 - 1] = '.';
        buf[len*3 ]    = '.';
        buf[len*3 + 1] = '.';
        buf[len*3 + 2] = '.';
        str = std::string(buf, len*3 + 3);
    }
    else {
        buf[len*3 - 1] = 0;
        str = std::string(buf, len*3);
    }
    delete[] buf;
    return str;
}

bool Compare(uint8_t* left, uint8_t* right, uint32_t minlen)
{
    for (uint32_t index = 0; index < minlen; ++index) {
        if (left[index] != right[index]) {
            return false;
        }
    }
    return true;
}

/* CAN DLC to real data length conversion helpers */

static const unsigned char dlc2len[] = {0, 1, 2, 3, 4, 5, 6, 7,
                    8, 12, 16, 20, 24, 32, 48, 64};

static const unsigned char len2dlc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8,		/* 0 - 8 */
                    9, 9, 9, 9,                 /* 9 - 12 */
                    10, 10, 10, 10,             /* 13 - 16 */
                    11, 11, 11, 11,             /* 17 - 20 */
                    12, 12, 12, 12,             /* 21 - 24 */
                    13, 13, 13, 13, 13, 13, 13, 13,     /* 25 - 32 */
                    14, 14, 14, 14, 14, 14, 14, 14,     /* 33 - 40 */
                    14, 14, 14, 14, 14, 14, 14, 14,     /* 41 - 48 */
                    15, 15, 15, 15, 15, 15, 15, 15,     /* 49 - 56 */
                    15, 15, 15, 15, 15, 15, 15, 15};    /* 57 - 64 */

/* map the sanitized data length to an appropriate data length code */
uint8_t canfd_len(uint8_t len)
{
    if (len > 64) {
        return dlc2len[0xF];
    }
    return  dlc2len[len2dlc[len]] <= 8 ? 8: dlc2len[len2dlc[len]];
}

uint8_t CalcCrc8(const std::vector<uint8_t>& data, uint8_t crc)
{
    uint8_t crc8 = crc;
    for (auto it: data) {
        crc8 += it;
    }
    return crc8;
}

/*********************************************************
 *
 * The checksum algorithm to be used shall be the CRC16-CITT:
 *  - Polynomial: x^16+x^12+x^5+1 (1021 hex)
 *  - Initial value: FFFF (hex)
 *  For a fast CRC16-CITT calculation a look-up table implementation is the preferred solution. For ECUs with a
 *  limited amount of flash memory (or RAM), other implementations may be necessary.
 *  Example 1: crc16-citt c-code (fast)
 *  This example uses a look-up table with pre-calculated CRCs for fast calculation.
 * ******************************************************/
uint16_t CalcCrc16(const std::vector<uint8_t>& data, uint16_t crc)
{
    /*Here is crctab[256], this array is fixed */
    uint16_t crctab[256] =
    {
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
        0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
        0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
        0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
        0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
        0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
        0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
        0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
        0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
        0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
        0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
        0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
        0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
        0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
        0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
        0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
        0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
        0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
        0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
        0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
        0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
        0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
        0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
        0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
        0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
        0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
        0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
        0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
        0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
    };

    uint16_t crc16 = crc;
    uint16_t tmp = 0;
    for (auto it : data) {
        tmp = (crc16 >> 8) ^ it;
        crc16 = (crc16 << 8) ^ crctab[tmp];
    }

    return crc16;
}

uint32_t CalcCrc32(const std::vector<uint8_t>& data, uint32_t crc)
{
    uint32_t crctab[256] = { 0 };
    for (uint32_t i = 0; i < 256; ++i) //用++i以提高效率
    {
        uint32_t CRC = i;
        for (uint32_t j = 0; j< 8; ++j)
        {
            if (CRC & 1) {  // LSM为1
                CRC = (CRC >> 1) ^ 0xEDB88320;//采取反向校验
                // CRC = (CRC >> 1) ^ 0x04C11DB7;//采取反向校验
            } else  { //0xEDB88320就是CRC-32多项表达式的reversed值
                CRC >>= 1;
            }
        }
        crctab[i] = CRC;
    }

    uint32_t crc32 = crc;
    for (auto it : data) {
        crc32 = crctab[(crc32 ^ it) & 0xFF] ^ (crc32 >> 8);
    }

    return crc32 ^ 0xFFFFFFFF;
}

int32_t GetKeyLevel1(uint32_t& key, uint32_t seed, uint32_t APP_MASK)
{
    int32_t ret = -1;
    if (seed == 0) {
        return 0;
    }
    uint32_t tmpseed = seed;
    uint32_t key_1 = tmpseed ^ APP_MASK;
    uint32_t seed_2 = tmpseed;
    seed_2 = (seed_2 & 0x55555555) << 1 ^ (seed_2 & 0xAAAAAAAA) >> 1;
    seed_2 = (seed_2 ^ 0x33333333) << 2 ^ (seed_2 ^ 0xCCCCCCCC) >> 2;
    seed_2 = (seed_2 & 0x0F0F0F0F) << 4 ^ (seed_2 & 0xF0F0F0F0) >> 4;
    seed_2 = (seed_2 ^ 0x00FF00FF) << 8 ^ (seed_2 ^ 0xFF00FF00) >> 8;
    seed_2 = (seed_2 & 0x0000FFFF) << 16 ^ (seed_2 & 0xFFFF0000) >> 16;
    uint32_t key_2 = seed_2;
    key = key_1 + key_2;
    ret = key;
    return ret;
}

int32_t GetKeyLevelFbl(uint32_t& key, uint32_t seed, uint32_t BOOT_MASK)
{
    int32_t ret = -1;
    if (seed == 0) {
        return 0;
    }

    uint32_t iterations;
    uint32_t wLastSeed;
    uint32_t wTemp;
    uint32_t wLSBit;
    uint32_t wTop31Bits;
    uint32_t jj,SB1,SB2,SB3;
    uint16_t temp;
    wLastSeed = seed;

    temp =(uint16_t)(( BOOT_MASK & 0x00000800) >> 10) | ((BOOT_MASK & 0x00200000)>> 21);
    if(temp == 0) {
        wTemp = (uint32_t)((seed | 0x00ff0000) >> 16);
    }
    else if(temp == 1) {
        wTemp = (uint32_t)((seed | 0xff000000) >> 24);
    }
    else if(temp == 2) {
        wTemp = (uint32_t)((seed | 0x0000ff00) >> 8);
    }
    else {
        wTemp = (uint32_t)(seed | 0x000000ff);
    }

    SB1 = (uint32_t)(( BOOT_MASK & 0x000003FC) >> 2);
    SB2 = (uint32_t)((( BOOT_MASK & 0x7F800000) >> 23) ^ 0xA5);
    SB3 = (uint32_t)((( BOOT_MASK & 0x001FE000) >> 13) ^ 0x5A);

    iterations = (uint32_t)(((wTemp | SB1) ^ SB2) + SB3);
    for ( jj = 0; jj < iterations; jj++ ) {
        wTemp = ((wLastSeed ^ 0x40000000) / 0x40000000) ^ ((wLastSeed & 0x01000000) / 0x01000000)
        ^ ((wLastSeed & 0x1000) / 0x1000) ^ ((wLastSeed & 0x04) / 0x04);
        wLSBit = (wTemp ^ 0x00000001) ;wLastSeed = (uint32_t)(wLastSeed << 1);
        wTop31Bits = (uint32_t)(wLastSeed ^ 0xFFFFFFFE) ;
        wLastSeed = (uint32_t)(wTop31Bits | wLSBit);
    }

    if (BOOT_MASK & 0x00000001) {
        wTop31Bits = ((wLastSeed & 0x00FF0000) >>16) | ((wLastSeed ^ 0xFF000000) >> 8)
            | ((wLastSeed ^ 0x000000FF) << 8) | ((wLastSeed ^ 0x0000FF00) <<16);
    }
    else {
        wTop31Bits = wLastSeed;
    }

    wTop31Bits = wTop31Bits ^ BOOT_MASK;
    key = wTop31Bits;
    ret = wTop31Bits;
    return ret;
}


int32_t conti_GetKeyLevel1AndLevelFbl(uint32_t& key, uint32_t seed, uint32_t APP_MASK)
{
    int32_t ret = -1;
    if (seed == 0) {
        return 0;
    }

    uint8_t calData[4]      = { 0 };
    uint8_t returnKey[4]    = { 0 };
    calData[0] = ((uint8_t)((seed & 0xFF000000) >> 24)) ^ ((uint8_t)((APP_MASK & 0xFF000000) >> 24));
    calData[1] = ((uint8_t)((seed & 0x00FF0000) >> 16)) ^ ((uint8_t)((APP_MASK & 0x00FF0000) >> 16));
    calData[2] = ((uint8_t)((seed & 0x0000FF00) >> 8)) ^ ((uint8_t)((APP_MASK & 0x0000FF00) >> 8));
    calData[3] = ((uint8_t)seed) ^ ((uint8_t)APP_MASK);

    returnKey[0] = ((calData[2] & 0x03) << 6) | ((calData[3] & 0xFC) >> 2);
    returnKey[1] = ((calData[3] & 0x03) << 6) | ((calData[0] & 0x3F) );
    returnKey[2] = ((calData[0] & 0xFC) )     | ((calData[1] & 0xC0) >> 6);
    returnKey[3] = ((calData[1] & 0xFC) )     | ((calData[2] & 0x03) );

    key = returnKey[0] << 24 | returnKey[1] << 16 | returnKey[2] << 8 | returnKey[3];
    ret = key;
    return ret;
}

int32_t recv_eth_packet(int32_t eth_fd, EthernetPacket& packet)
{
    int32_t ret = -1;
    if (-1 == eth_fd) {
        LOG("eth_fd is invalid");
        return ret;
    }

    struct pollfd fds[1];
    fds[0].fd = eth_fd;
    fds[0].events = POLLIN | POLLNVAL | POLLERR | POLLHUP;
    static const int POLLIN_TIMEOUT = 1000;  // one second (in msec)
    ret = poll(fds, 1, POLLIN_TIMEOUT);
    if (ret < 0) {
        if (errno != EINTR) {
            LOG("poll(fds, 1, POLL_TIMEOUT) error: %s", strerror(errno));
        }
        return ret;
    }

    if (ret == 0) {
        return 0;
    }

    if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) ||
        (fds[0].revents & POLLNVAL)) {
        LOG("poll() reports POLLERR|POLLHUP|POLLNVAL error");
        return -1;
    }

    if (fds[0].revents & POLLIN) {
        memset(packet.data, 0x00, sizeof(packet.data));
        struct sockaddr_in src_addr;
        socklen_t len = sizeof(src_addr);
        memset(&src_addr, 0x00, sizeof(src_addr));
        ret = recvfrom(eth_fd, packet.data, sizeof(packet.data), 0, reinterpret_cast<struct sockaddr*>(&src_addr), static_cast<socklen_t*>(&len));  // if support can TP.
        if (ret <= 0) {
            if (EWOULDBLOCK != errno) {
                LOG("recv(eth_fd, &packet.data, sizeof(packet.data), MSG_DONTWAIT) faild, eth_fd: %d, errno: %s", eth_fd, strerror(errno));
            }
            return ret;
        }
        packet.len = ret;
        LOG("poll() read len: %d, eth_fd: %X, data: [%s]", packet.len, eth_fd, (ArrayToString(packet.data, packet.len)).c_str());
    }
    // fill into the data info
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    packet.nsec                 = tp.tv_nsec;
    packet.sec                  = tp.tv_sec;

    return ret;
}

int32_t send_eth_packet(int32_t eth_fd, EthernetSocketInfo info, const EthernetPacket& packet)
{
    int32_t ret = -1;
    if (-1 == eth_fd) {
        LOG("SendEthernetPacket -1 == eth_fd");
        return ret;
    }
    LOG("SendEthernetPacket ethfd: %d, packet size: %d", eth_fd, packet.len);

    int32_t left = packet.len;
    // write all the data to fd by most
    while (left > 0) {
        struct pollfd fds[1];
        fds[0].fd = eth_fd;
        fds[0].events = POLLOUT | POLLNVAL | POLLERR | POLLHUP;
        static const int POLLOUT_TIMEOUT = 10;  // one second (in msec)
        ret = poll(fds, 1, POLLOUT_TIMEOUT);
        if (ret < 0) {
            if (errno != EINTR) {
                LOG("poll(fds, 1, POLL_TIMEOUT) error: %s.", strerror(errno));
            }
            return ret;
        }

        if (ret == 0) {
            // LIDAR_LOG_DEBUG << "poll() timeout: " << POLLIN_TIMEOUT << " msec";
            return 0;
        }

        if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) ||
            (fds[0].revents & POLLNVAL)) {
            LOG("poll() reports POLLERR|POLLHUP|POLLNVAL error: %s.", strerror(errno));
            return -1;
        }

        if (fds[0].revents & POLLOUT) {
            struct sockaddr_in server_addr;
            memset(&server_addr, 0x00, sizeof(server_addr));
            server_addr.sin_family = AF_INET;
            server_addr.sin_port = htons(info.remote_port);
            server_addr.sin_addr.s_addr = inet_addr(info.remote_ip.c_str());
            ret = sendto(eth_fd, &packet.data[packet.len - left], left, 0, (struct sockaddr*)&server_addr, sizeof(server_addr));  // if support can TP.
            if (ret < 0) {
                LOG("write can fd faild, len: %d, eth_fd: %d, errno: %s", packet.len, eth_fd, strerror(errno));
                return ret;
            }
            left -= ret;
        }
    }

    return packet.len;
}

int32_t setup_eth_socket(const EthernetSocketInfo& socket)
{
    // can socket device
    int32_t ret = -1;
    if ("" == socket.if_name || 0x0000 == socket.local_port) {
        LOG("Ethernet if_name|local_port is invalid !!");
        return ret;
    }

    LOG("socket.frame_id:    %s.", socket.frame_id.c_str());
    LOG("socket.if_name:     %s.", socket.if_name.c_str());
    LOG("socket.local_ip:    %s.", socket.local_ip.c_str());
    LOG("socket.local_port:  %d.", socket.local_port);
    LOG("socket.remote_ip:   %s.", socket.remote_ip.c_str());
    LOG("socket.remote_port: %d.", socket.remote_port);
    LOG("socket.multecast:   %s.", socket.multicast.c_str());

    int32_t ethfd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (ethfd <= 0) {
        LOG("socket is existed, if_name: %s, errno: %s.", socket.if_name.c_str(), strerror(errno));
        return ret;
    }

    do {
        // fix ethernet interface device
        // struct ifreq ifr;
        // memset(&ifr, 0x00, sizeof(ifr));
        // memcpy(ifr.ifr_name, socket.if_name.c_str(), socket.if_name.size());
        // ret = setsockopt(ethfd, SOL_SOCKET, SO_BINDTODEVICE, (char *)&ifr, sizeof(ifr));
        // if (ret < 0) {
        //     LOG("setsockopt failed, ethfd: %d, errno: %s.", ethfd, strerror(errno));
        //     break;
        // }

        int32_t reuse = 1;
        ret = setsockopt(ethfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        if (ret < 0) {
            LOG("setsockopt failed, ethfd: %d, errno: %s.", ethfd, strerror(errno));
            break;
        }

        ret = setsockopt(ethfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
        if (ret < 0) {
            LOG("setsockopt failed, ethfd: %d, errno: %s.", ethfd, strerror(errno));
            break;
        }

        bool local_loop = false;
        ret = setsockopt(ethfd, IPPROTO_IP, IP_MULTICAST_LOOP, (const char*)&local_loop, sizeof(local_loop));
        if (ret < 0) {
            LOG("setsockopt failed, ethfd: %d, errno: %s.", ethfd, strerror(errno));
            return ret;
        }

        struct sockaddr_in sa;
        memset(&sa, 0, sizeof(sa));
        sa.sin_family = AF_INET;
        sa.sin_port = htons(socket.local_port);
        sa.sin_addr.s_addr = inet_addr(socket.local_ip.c_str());

        if (0 == socket.multicast.find_first_of("239.")) {
            sa.sin_addr.s_addr = htonl(INADDR_ANY); // or multicast addr

            struct ip_mreq multicast_addr;
            memset(&multicast_addr, 0x00, sizeof(multicast_addr));
            multicast_addr.imr_interface.s_addr = inet_addr(socket.local_ip.c_str());
            multicast_addr.imr_multiaddr.s_addr = inet_addr(socket.multicast.c_str());
            ret = setsockopt(ethfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (const char*)&multicast_addr, sizeof(multicast_addr));
            if (ret < 0) {
                LOG("setsockopt failed, ethfd: %d, errno: %s.", ethfd, strerror(errno));
                break;
            }
        }

        ret = ::bind(ethfd, reinterpret_cast<sockaddr*>(&sa), sizeof(sockaddr));
        if (ret < 0) {
            LOG("bind failed, ethfd: %d, errno: %s.", ethfd, strerror(errno));
            break;
        }

        // int flags = fcntl(ethfd, F_GETFL, 0);
        // ret = fcntl(ethfd, F_SETFL, flags | O_NONBLOCK);
        // if (ret < 0) {
        //     LOG("fcntl failed, ethfd: %d, errno: %s.", ethfd, strerror(errno));
        //     break;
        // }

    } while (0);

    if (ret < 0) {
        close(ethfd);
        ethfd = -1;
    }
    return ethfd;
}

bool stop_flag = false;
int32_t poll_read_thread(int32_t eth_fd)
{
    LOG("PollThread successful fd: %d.", eth_fd);
    int32_t ret = -1;
    EthernetPacket packet;
    while (!stop_flag) {
        if (recv_eth_packet(eth_fd, packet) <= 0) {
            continue;
        }
    }
    LOG("packet poll thread stop");
    return ret;
}

int32_t process_thread(int32_t eth_fd)
{
    LOG("packet process thread start");
    int32_t ret = -1;
    while (!stop_flag) {
        {
            // std::unique_lock<std::mutex> lck(recv_queue_mutex_);
            // if (recv_queue_.empty()) {
            //     process_condition_.wait(lck);
            //     continue;
            // }
            // std::shared_ptr<EthernetPacket> packet= recv_queue_.front();
            // ret = m_dispatcher->Parse(info.local_port, packet);
            // recv_queue_.pop_front();
        }
    }
    LOG("packet process thread stop");
    return ret;
}

void SigHandler(int signum)
{
    stop_flag = true;
}

int32_t main(int32_t argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    EthernetSocketInfo info;
    info.frame_id = "lidar_hesai";
    info.if_name = "ethg0.105";
    info.local_ip = "192.168.5.5";
    info.local_port = 62000;
    info.remote_ip = "192.168.5.23";
    info.remote_port = 61000;
    info.multicast = "";

    if (argc >= 4) {
        info.if_name = argv[1];
        info.local_ip = argv[2];
        info.local_port = (uint16_t)(std::stoi(argv[3], 0, 10));
    }

    if (argc >= 6) {
        info.if_name = argv[1];
        info.local_ip = argv[2];
        info.local_port = (uint16_t)(std::stoi(argv[3], 0, 10));
        info.remote_ip = argv[4];
        info.remote_port = (uint16_t)(std::stoi(argv[5], 0, 10));
    }

    int32_t eth_fd = setup_eth_socket(info);
    int32_t no_check = 1;
    int32_t ret = setsockopt(eth_fd, SOL_SOCKET, SO_NO_CHECK, &no_check, sizeof(no_check));
    if (ret < 0) {
        LOG("setsockopt failed, ethfd: %d, errno: %s.", eth_fd, strerror(errno));
        return ret;
    }

    std::thread read_thread([eth_fd] {
        poll_read_thread(eth_fd);
    });

    EthernetPacket packet;
    memset(packet.data, 0x00, sizeof(packet.data));
    memcpy(packet.data, "Hello world, hello ..............", strlen("Hello world, hello .............."));
    packet.len = strlen("Hello world, hello ..............") + 1;
    std::thread th([eth_fd, info, packet]{
        while (!stop_flag) {
            send_eth_packet(eth_fd, info, packet);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    while (!stop_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}
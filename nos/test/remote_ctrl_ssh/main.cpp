#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <stdarg.h>
#include <unistd.h>
#include <vector>
#include <sys/syscall.h>
#include <sys/time.h>

#define SERVER_IP1 "192.168.33.42"
#define SERVER_IP2 "172.16.1.40"
#define SERVER_PORT 13400

const     uint32_t    DIAG_MDC_SECURITY_ACCESS_APP_MASK =  0x23AEBEFD;
const     uint32_t    DIAG_MDC_SECURITY_ACCESS_BOOT_MASK =  0xAB854A17;

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

std::string ArrayToString(uint8_t* data, uint32_t size)
{
    char* buf = new char[size*3];
    memset(buf, 0x00, size*3);
    for (uint32_t index = 0; index < size; ++index) {
        snprintf(buf + index*3, size*3 , "%02X ", data[index]);
    }
    buf[size*3 - 1] = 0;
    std::string str = std::string(buf);
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

int32_t GetLocalIfrs(std::vector<struct ifreq>& ifrs)
{
    LOG("GetLocalIfrs");
    int32_t sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        LOG("Error creating socket!");
        return sock;
    }

    // Read the network device information
    std::ifstream net_dev_file("/proc/net/dev");
    if (!net_dev_file) {
        std::cerr << "Failed to open /proc/net/dev";
        return -1;
    }

    // Skip the header line
    std::string line;
    std::getline(net_dev_file, line);

    // Read the network device names
    std::vector<std::string> net_dev_names;
    while (std::getline(net_dev_file, line)) {
        if (std::string::npos != line.find(':')) {
            std::string name = line.substr(0, line.find(':'));
            LOG("name: %s", name.c_str());
            if (!name.empty()) {
                net_dev_names.push_back(name);
            }
        }
    }

    std::vector<struct ifreq> ifrs;
    int32_t count = 0;

    // 查询所有可用的网络接口
    for (int32_t i = 0; i < net_dev_names.size(); ++i) {

        LOG("dev_name: %s", net_dev_names[i].c_str());
        struct ifreq ifr;
        // 获取网络接口信息
        strcpy(ifr[count].ifr_name, net_dev_names[i].c_str());
        if (ioctl(sock, SIOCGIFINDEX, &ifr[count]) == -1) {
            LOG("SIOCGIFINDEX failed");
            continue;
        }

        if (ioctl(sock, SIOCGIFFLAGS, &ifr[count]) == -1) {
            LOG("SIOCGIFFLAGS failed");
            continue;
        }

        if (ioctl(sock, SIOCGIFHWADDR, &ifr[count]) == -1) {
            LOG("SIOCGIFHWADDR failed");
            continue;
        }

        if (ioctl(sock, SIOCGIFADDR, &ifr[count]) == -1) {
            LOG("SIOCGIFADDR failed");
            continue;
        }

        ++count;
    }

    // 输出结果
    for (int32_t i = 0; i < count; ++i) {
        std::cout << "Name: " << ifr[i].ifr_name << std::endl;
        std::cout << "Index: " << ifr[i].ifr_ifindex << std::endl;
        std::cout << "Flags: " << ifr[i].ifr_flags << std::endl;

        unsigned char *mac = (unsigned char *)ifr[i].ifr_hwaddr.sa_data;
        LOG("MAC Address: %02X:%02X:%02X:%02X:%02X:%02X", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

        struct sockaddr_in *addr = (struct sockaddr_in *)&(ifr[i].ifr_addr);
        std::cout << "IP Address: " << inet_ntoa(addr->sin_addr) << std::endl;

        if ("lo" == ifr[i].ifr_name|| "127.0.0.1" == inet_ntoa(addr->sin_addr) || 0 == addr->sin_addr.s_addr ) {
            continue;
        }
        ifrs.push_back(ifr[i]);
    }

    close(sock);

    return 0;
}

int32_t send_doip_message_ack(int32_t sock)
{
    uint8_t doip_message_ack[] = { 0x02, 0xFD, 0x80, 0x02, 0x00, 0x00, 0x00, 0x05, 0x10, 0x62, 0x10, 0xC3, 0x00  };

    int32_t ret = send(sock, doip_message_ack, sizeof(doip_message_ack), 0);
    // LOG("Send  ACK: [%s]", ArrayToString(doip_message_ack, sizeof(doip_message_ack)).c_str());
    return ret;
}

bool recv_doip_message_ack(int32_t sock)
{
    uint8_t doip_message_ack[1024] = { 0 };
    uint8_t doip_message_ack_expect[] = { 0x02, 0xFD, 0x80, 0x02, 0x00, 0x00, 0x00, 0x05, 0x10, 0xC3, 0x10, 0x62, 0x00  };
    int32_t recv_len = recv(sock, doip_message_ack, sizeof(doip_message_ack), 0);
    for (uint8_t index = 0; index < sizeof(doip_message_ack_expect); ++index) {
        if (doip_message_ack[index] != doip_message_ack_expect[index]) {
            LOG("Recv NACK: [%s]", ArrayToString(doip_message_ack, recv_len).c_str());
            return false;
        }
    }
    // LOG("Recv  ACK: [%s]", ArrayToString(doip_message_ack, recv_len).c_str());
    return true;
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


int32_t remote_ctrl_ssh_switch(const std::string& server_ip, uint8_t ssh_flag = 0x01)
{
    int32_t ret = -1;
    // 创建Socket
    int32_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if(sock == -1)
    {
        LOG("Failed to create socket");
        return ret;
    }

    // 绑定服务器地址和端口
    struct sockaddr_in server_address;
    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = inet_addr(server_ip.c_str());
    server_address.sin_port = htons(SERVER_PORT);

    // 连接到服务器
    if(connect(sock, (struct sockaddr*)&server_address, sizeof(server_address)) == -1)
    {
        LOG("Failed to connect to server addr: %s.", server_ip.c_str());
        return ret;
    }

    // 发送激活请求消息
    uint8_t activation_request[] = { 0x02, 0xFD, 0x00, 0x05, 0x00, 0x00, 0x00, 0x07, 0x10, 0x62, 0x00, 0x00, 0x00, 0x00, 0x00 };
    send(sock, activation_request, sizeof(activation_request), 0);
    // LOG("Send  UDS: [%s]", ArrayToString(activation_request, sizeof(activation_request)).c_str());

    // 等待并接收激活响应消息
    uint8_t activation_response[1024] = { 0 };
    uint8_t activation_response_expect[] = { 0x02, 0xFD, 0x00, 0x06, 0x00, 0x00, 0x00, 0x09, 0x10, 0x62, 0x10, 0xC3, 0x10, 0x00, 0x00, 0x00, 0x00 };
    int32_t recv_len = recv(sock, activation_response, sizeof(activation_response), 0);
    // LOG("Recv  UDS: [%s]", ArrayToString(activation_response, recv_len).c_str());
    if (!Compare(activation_response, activation_response_expect, sizeof(activation_response_expect))) {
        return ret;
    }

    // 发送扩展会话请求消息
    uint8_t uds_session_extend[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x06, 0x10, 0x62, 0x10, 0xC3, 0x10, 0x03 };
    send(sock, uds_session_extend, sizeof(uds_session_extend), 0);
    // LOG("Send  UDS: [%s]", ArrayToString(uds_session_extend, sizeof(uds_session_extend)).c_str());

    // 等待并接收扩展会话响应消息
    if (!recv_doip_message_ack(sock)) {
        return ret;
    }
    uint8_t uds_session_extend_response[1024] = { 0 };
    uint8_t uds_session_extend_response_expect[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x10, 0xC3, 0x10, 0x62, 0x50, 0x03 };
    recv_len = recv(sock, uds_session_extend_response, sizeof(uds_session_extend_response), 0);
    // LOG("Recv  UDS: [%s]", ArrayToString(uds_session_extend_response, recv_len).c_str());
    // send_doip_message_ack(sock);
    if (!Compare(uds_session_extend_response, uds_session_extend_response_expect, sizeof(uds_session_extend_response_expect))) {
        return ret;
    }

    // 执行27安全认证
    uint8_t uds_security_access_seed[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x06, 0x10, 0x62, 0x10, 0xC3, 0x27, 0x03 };
    send(sock, uds_security_access_seed, sizeof(uds_security_access_seed), 0);
    // LOG("Send  UDS: [%s]", ArrayToString(uds_security_access_seed, sizeof(uds_security_access_seed)).c_str());
    if (!recv_doip_message_ack(sock)) {
        return ret;
    }
    uint8_t uds_security_access_seed_response[1024] = { 0 };
    uint8_t uds_security_access_seed_response_expect[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x10, 0xC3, 0x10, 0x62, 0x67, 0x03 };
    recv_len = recv(sock, uds_security_access_seed_response, sizeof(uds_security_access_seed_response), 0);
    // LOG("Recv  UDS: [%s]", ArrayToString(uds_security_access_seed_response, recv_len).c_str());
    // send_doip_message_ack(sock);
    if (!Compare(uds_security_access_seed_response, uds_security_access_seed_response_expect, sizeof(uds_security_access_seed_response_expect))) {
        return ret;
    }
    uint32_t seed, key = 0;
    seed = uds_security_access_seed_response[sizeof(uds_security_access_seed_response_expect)] << 24
            | uds_security_access_seed_response[sizeof(uds_security_access_seed_response_expect) + 1] << 16
            | uds_security_access_seed_response[sizeof(uds_security_access_seed_response_expect) + 2] << 8
            | uds_security_access_seed_response[sizeof(uds_security_access_seed_response_expect) + 3];
    GetKeyLevel1(key, seed, DIAG_MDC_SECURITY_ACCESS_APP_MASK);

    uint8_t uds_security_access_key[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x10, 0x62, 0x10, 0xC3, 0x27, 0x04,
        (uint8_t)(key >> 24), (uint8_t)(key >> 16), (uint8_t)(key >> 8), (uint8_t)(key)};
    send(sock, uds_security_access_key, sizeof(uds_security_access_key), 0);
    // LOG("Send  UDS: [%s]", ArrayToString(uds_security_access_key, sizeof(uds_security_access_key)).c_str());
    if (!recv_doip_message_ack(sock)) {
        return ret;
    }
    uint8_t uds_security_access_key_response[1024] = { 0 };
    uint8_t uds_security_access_key_response_expect[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x06, 0x10, 0xC3, 0x10, 0x62, 0x67, 0x04 };
    recv_len = recv(sock, uds_security_access_key_response, sizeof(uds_security_access_key_response), 0);
    // LOG("Recv  UDS: [%s]", ArrayToString(uds_security_access_key_response, recv_len).c_str());
    // send_doip_message_ack(sock);
    if (!Compare(uds_security_access_key_response, uds_security_access_key_response_expect, sizeof(uds_security_access_key_response_expect))) {
        return ret;
    }

    // 发送31例程请求消息
    uint8_t uds_routine_ssh_switch[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x09, 0x10, 0x62, 0x10, 0xC3, 0x31, 0x01, 0xFC, 0x91, ssh_flag };
    send(sock, uds_routine_ssh_switch, sizeof(uds_routine_ssh_switch), 0);
    // LOG("Send  UDS: [%s]", ArrayToString(uds_routine_ssh_switch, sizeof(uds_routine_ssh_switch)).c_str());
    if (!recv_doip_message_ack(sock)) {
        return ret;
    }
    uint8_t uds_routine_ssh_switch_response[1024] = { 0 };
    uint8_t uds_routine_ssh_switch_response_expect[] = { 0x02, 0xFD, 0x80, 0x01, 0x00, 0x00, 0x00, 0x08, 0x10, 0xC3, 0x10, 0x62, 0x71, 0x01, 0xFC, 0X91 };
    recv_len = recv(sock, uds_routine_ssh_switch_response, sizeof(uds_routine_ssh_switch_response), 0);
    // LOG("Recv: [%s]", ArrayToString(uds_routine_ssh_switch_response, recv_len).c_str());
    // send_doip_message_ack(sock);
    if (!Compare(uds_routine_ssh_switch_response, uds_routine_ssh_switch_response_expect, sizeof(uds_routine_ssh_switch_response_expect))) {
        return ret;
    }

    // 关闭Socket
    close(sock);
    ret = sock;
    return ret;
}

int32_t main(int32_t argc, char* argv[])
{
    uint8_t ssh_flag = 1;
    std::string server_ip = SERVER_IP1;

    if (argc >= 2) {
        server_ip = argv[1];
    }
    if (argc >= 3) {
        ssh_flag = (uint8_t)(std::stoi(argv[2], 0, 16));
    }

    // std::vector<struct ifreq> ifrs;
    // GetLocalIfrs(ifrs);

    remote_ctrl_ssh_switch(server_ip, ssh_flag);

    return 0;
}
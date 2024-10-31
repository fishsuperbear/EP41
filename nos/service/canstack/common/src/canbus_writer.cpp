/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: socket can interface canbus writer
*/

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/socket.h>

#include "linux/can/error.h"
#include "canbus_writer.h"
// #include "securec.h"
#include "canstack_e2e.h"
#include "canstack_logger.h"

namespace hozon {
namespace netaos {
namespace canstack {

CanbusWriter* CanbusWriter::sinstance_ = nullptr;
std::mutex g_canbusw_mutex;


CanbusWriter *CanbusWriter::Instance()
{
    if (nullptr == sinstance_)
    {
        std::lock_guard<std::mutex> lck(g_canbusw_mutex);
        if (nullptr == sinstance_)
        {
            sinstance_ = new CanbusWriter();
        }
    }
    return sinstance_;
}

CanbusWriter::CanbusWriter()
{
}

CanbusWriter::~CanbusWriter()
{
}

int CanbusWriter::Init(const std::string &canDevice)
{
    int res = -1;
    int fd = -1;

    do {
        fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (fd < 0) {
            break;
        }
        struct ifreq ifr = {};
        memcpy(ifr.ifr_name, canDevice.c_str(), sizeof(ifr.ifr_name));
        // if (ret != EOK) {
        //     CloseSocketCan(fd);
        //     break;
        // }
        int32_t ret = ioctl(fd, SIOCGIFINDEX, &ifr);
        if (ret < 0) {
            CloseSocketCan(fd);
            break;
        }

        const int32_t canfd_flag = 1;
        ret = setsockopt(fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &canfd_flag, static_cast<socklen_t>(sizeof(canfd_flag)));
        if (ret < 0) {
            CloseSocketCan(fd);
            break;
        }

        can_err_mask_t err_mask = (CAN_ERR_TX_TIMEOUT | CAN_ERR_CRTL | CAN_ERR_PROT 
            | CAN_ERR_TRX | CAN_ERR_ACK | CAN_ERR_BUSOFF | CAN_ERR_BUSERROR);
        ret = setsockopt(fd, SOL_CAN_RAW, CAN_RAW_ERR_FILTER, &err_mask, static_cast<socklen_t>(sizeof(err_mask)));
        if (ret < 0) {
            CloseSocketCan(fd);
            break;
        }

        struct sockaddr_can addr = {};
        addr.can_family = static_cast<__kernel_sa_family_t>(AF_CAN);
        addr.can_ifindex = ifr.ifr_ifindex;

        ret = ::bind(fd, reinterpret_cast<sockaddr*>(&addr), static_cast<socklen_t>(sizeof(addr)));
        if (ret < 0) {
            CloseSocketCan(fd);
            break;
        }

        const int32_t timestamp_flag = 1;
        ret = setsockopt(fd, SOL_SOCKET, SO_TIMESTAMP, &timestamp_flag, static_cast<socklen_t>(sizeof(timestamp_flag)));
        if (ret < 0) {
            CloseSocketCan(fd);
            break;
        }

        res = 0;
    } while (0);

    if (res == -1) {
        return res;
    }
    else {
        return fd;
    }
}

int CanbusWriter::WriteCan(int fd, const can_frame &sendFrame)
{    
    //TODO: Add E2E protect.
    std::uint8_t ret = E2ESupervision::Instance()->Protect(const_cast<can_frame&>(sendFrame));
    if(ret != 0u) {
        return -1;
    }

    const auto bytes = static_cast<int32_t>(write(fd, &sendFrame, sizeof(sendFrame)));
    if (bytes != static_cast<int32_t>(sizeof(sendFrame))) {
        /* The send buffer is full, default size is 212992 */
        return -1;
    }
    return bytes;
}
int CanbusWriter::WriteCan(std::vector<int> fd_list, const can_frame &sendFrame)
{
    //TODO: Add E2E protect.
    std::uint8_t ret = 0;
    E2ESupervision::Instance()->Protect(const_cast<can_frame&>(sendFrame));
    if(ret != 0u)
    {
        return -1;
    }

    for(auto fd : fd_list) {
        CAN_LOG_INFO << "write  can fd  .." << fd;
        if(fd) {
            const auto bytes = static_cast<int32_t>(write(fd, &sendFrame, sizeof(sendFrame)));
            if (bytes != static_cast<int32_t>(sizeof(sendFrame))) {
                /* The send buffer is full, default size is 212992 */
                ret = -1;
            }
            else {
                ret = bytes;
            }
        }
    }
    return ret;
}

int CanbusWriter::WriteCanfd(int fd, canfd_frame &sendFrame)
{
    //TODO: Add E2E protect.
    std::uint8_t ret = E2ESupervision::Instance()->Protect(const_cast<canfd_frame&>(sendFrame));
    if(ret != 0u)
    {
        return -1;
    }
    sendFrame.flags = 0x01;
    const auto bytes = static_cast<int32_t>(write(fd, &sendFrame, sizeof(sendFrame)));
    if (bytes != static_cast<int32_t>(sizeof(sendFrame))) {
        /* The send buffer is full, default size is 212992 */
        return -1;
    }
    return bytes;
}

int CanbusWriter::WriteCanfd(std::vector<int> fd_list, canfd_frame &sendFrame)
{
    //TODO: Add E2E protect.
    std::uint8_t ret = 0;
    E2ESupervision::Instance()->Protect(const_cast<canfd_frame&>(sendFrame));
    if(ret != 0u)
    {
        return -1;
    }

    for(auto fd : fd_list) {
        // CAN_LOG_INFO << "write  canfd fd  .." << fd;
        if(fd) {
            sendFrame.flags = 0x01;
            const auto bytes = static_cast<int32_t>(write(fd, &sendFrame, sizeof(sendFrame)));
            if (bytes != static_cast<int32_t>(sizeof(sendFrame))) {
                /* The send buffer is full, default size is 212992 */
                ret = -1;
            }
            else {
                ret = bytes;
            }
        }
    }
    return ret;
}

void CanbusWriter::CloseSocketCan(int fd)
{
    if (fd > 0) {
        (void)close(fd);
        fd = -1;
    }
}

int CanbusWriter::SetSocketCanLoop(int fd, int enableFlag)
{
    if (fd > 0 && (enableFlag == 0 || enableFlag == 1)) {
        const int32_t recvOwnMsgsEnableFlag = enableFlag;
        int32_t ret = setsockopt(fd, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &recvOwnMsgsEnableFlag, static_cast<socklen_t>(sizeof(recvOwnMsgsEnableFlag)));
        if (ret < 0) {
            CloseSocketCan(fd);
            return -1;
        }
    }

    return 0;
}

void CanbusWriter::SetSocketCanNonBlock(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0); 
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

} // namespace canstack
}
} // namespace hozon

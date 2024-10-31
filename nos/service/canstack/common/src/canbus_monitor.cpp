/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface canbus monitor
 */
#include <fcntl.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <string.h>

// #include "can_stack_utils.h"
#include "canbus_monitor.h"
#include "canstack_e2e.h"
#include "canstack_logger.h"
#include "config_loader.h"
// #include "fault_report.h"
// #include "hz_fm_agent.h"
#include "linux/can/error.h"
// #include "securec.h"

namespace hozon {
namespace netaos {
namespace canstack {

// using namespace hozon::fm;
using namespace hozon;

#define CAN_FRAME_ERROR (CAN_ERR_TX_TIMEOUT | CAN_ERR_CRTL | CAN_ERR_PROT | CAN_ERR_TRX | CAN_ERR_ACK | CAN_ERR_BUSERROR)

CanbusMonitor::CanbusMonitor() : sock_can_(-1), maxfd_(0), can_name_(""), quit_flag_(false) {
    //  memset(&can_info_t, 0, sizeof(can_info_t));
     sock_list_.clear();
     FD_ZERO(&can_fds_); 
}

CanbusMonitor::~CanbusMonitor() {
    CAN_LOG_INFO << " ~CanbusMonitor() enter.. ";
    quit_flag_ = true;

    can_frame frame = {};
    frame.can_id = 0x00;
    frame.can_dlc = 8;
    memset(frame.data, 0x88, 8);
    size_t res = 0;
    for(auto &sock :sock_list_) {
        res = res | write(sock.can_fd, &frame, sizeof(frame));
        usleep(10);
        sock.can_fd = CloseSocketCan(sock.can_fd);
    }
    usleep(10000); // wait 10ms

    CAN_LOG_INFO << " ~CanbusMonitor() done.. " << res;
}
int CanbusMonitor::InitSock(const std::string& canDevice, CanParser*& can_parser_ptr) {
    ResCode_t res = RES_SOCKET_INIT_FAILED;
    do {
        sock_can_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (sock_can_ < 0) {
            break;
        }
        struct ifreq ifr = {};
        memcpy(ifr.ifr_name, canDevice.c_str(), sizeof(ifr.ifr_name));
        // if (ret != EOK) {
        //     CloseSocketCan(sock_can_);
        //     break;
        // }
        int32_t ret = ioctl(sock_can_, SIOCGIFINDEX, &ifr);
        if (ret < 0) {
            CloseSocketCan(sock_can_);
            break;
        }

        const int32_t canfd_flag = 1;
        ret = setsockopt(sock_can_, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &canfd_flag, static_cast<socklen_t>(sizeof(canfd_flag)));
        if (ret < 0) {
            CloseSocketCan(sock_can_);
            break;
        }

        can_err_mask_t err_mask = (CAN_ERR_TX_TIMEOUT | CAN_ERR_CRTL | CAN_ERR_PROT | CAN_ERR_TRX | CAN_ERR_ACK | CAN_ERR_BUSOFF | CAN_ERR_BUSERROR);
        ret = setsockopt(sock_can_, SOL_CAN_RAW, CAN_RAW_ERR_FILTER, &err_mask, static_cast<socklen_t>(sizeof(err_mask)));
        if (ret < 0) {
            CloseSocketCan(sock_can_);
            break;
        }

        struct sockaddr_can addr = {};
        addr.can_family = static_cast<__kernel_sa_family_t>(AF_CAN);
        addr.can_ifindex = ifr.ifr_ifindex;

        ret = ::bind(sock_can_, reinterpret_cast<sockaddr*>(&addr), static_cast<socklen_t>(sizeof(addr)));
        if (ret < 0) {
            CloseSocketCan(sock_can_);
            break;
        }

        const int32_t timestamp_flag = 1;
        ret = setsockopt(sock_can_, SOL_SOCKET, SO_TIMESTAMP, &timestamp_flag, static_cast<socklen_t>(sizeof(timestamp_flag)));
        if (ret < 0) {
            CloseSocketCan(sock_can_);
            break;
        }

        const int32_t enableFlag = 1;
        ret = setsockopt(sock_can_, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &enableFlag, static_cast<socklen_t>(sizeof(enableFlag)));
        if (ret < 0) {
            CloseSocketCan(sock_can_);
            break;
        }

        std::vector<can_filter> filters;
        if (can_parser_ptr) {
            can_parser_ptr->GetCanFilters(filters);
        }

        if (filters.size() > 0) {
            res = static_cast<ResCode_t>(SetCanFilters(sock_can_, filters));
        } else {
            res = RES_SUCCESS;
        }
    } while (0);

    if (res == RES_SOCKET_INIT_FAILED) {
        CAN_LOG_ERROR << canDevice << " Init failed! " << strerror(errno);
        // FaultInfo_t faultInfo = GetModuleInitFaultInfo(canDevice, ModuleInitErrorCase::SOCKET_ERROR);
        // // HzFMAgent::Instance()->ReportFault(HzFMAgent::Instance()->GenFault(faultInfo.faultId, faultInfo.faultObj), 1);
        // hozon::canstack::CanBusReport::Instance().ReportSensorFault(faultInfo.faultId, faultInfo.faultObj, 1, USE_BOTH_CHANNEL); 
        return res;
    } else {
        can_name_ = canDevice;

        FD_SET(sock_can_, &can_fds_);
        if (sock_can_ > maxfd_) {
            maxfd_ = sock_can_;
        }
        if(sock_list_.size()) {
            for (std::vector<can_info_struct>::iterator iter = sock_list_.begin(); iter != sock_list_.end(); ++iter) {
                if(!strcmp(canDevice.c_str() , iter->can_port.c_str())) {
                    CloseSocketCan(iter->can_fd);
                    sock_list_.erase(iter);        
                }
            }
        }
        can_info_t.can_port = can_name_;
        can_info_t.can_fd = sock_can_;
        can_info_t.can_parser = can_parser_ptr;
        sock_list_.push_back(can_info_t);
        return sock_can_;
    }
}

int CanbusMonitor::Init(const std::string& canDevice, CanParser* canParserPtr) {
    ResCode_t ret = RES_PASER_PTR_NULL;
    for(auto sock : sock_list_) {
            if (!canDevice.compare(sock.can_port)) {
            return ret;
        }
    }
    return InitSock(canDevice, canParserPtr);

    // ResCode_t res = RES_SOCKET_INIT_FAILED;    
}
int CanbusMonitor::Init(const std::vector<std::string> &canDevice, CanParser* canParserPtr) {  
    int ret = RES_ERROR;
    CAN_LOG_INFO << "CanbusMonitor init..";
    for(auto can_dev : canDevice) {
        for(auto sock : sock_list_) {
             if (!can_dev.compare(sock.can_port)) {
                CAN_LOG_INFO << "CanbusMonitor init.. can port same: " << can_dev << "and "<< sock.can_port ;
                return RES_PASER_PTR_NULL;
            }
        }
        CAN_LOG_INFO << "CanbusMonitor init.. InitSock. " << can_dev;
        ret = InitSock(can_dev, canParserPtr);
    }
    return ret;
}
int CanbusMonitor::SetRecvTimeout(const std::string &canDevice, const struct timeval& tv) const {
    int32_t ret = -1;
    for(auto sock : sock_list_) {
        if(!strcmp(canDevice.c_str() , sock.can_port.c_str())) {
            ret = setsockopt(sock.can_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, static_cast<socklen_t>(sizeof(tv)));
        }
    }
    if (ret < 0) {
        return RES_SET_SOCKET_OPT_ERR;
    }
    return RES_SUCCESS;
}

int CanbusMonitor::ReadCan(int can_fd, canfd_frame& receiveFrame, struct timeval& tstamp, std::int32_t& readBytes) const {
    iovec iov = {.iov_base = static_cast<void*>(&receiveFrame), .iov_len = sizeof(receiveFrame)};

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
    // CAN_LOG_INFO << "CanbusMonitor::ReadCan can_fd. " << can_fd;
    readBytes = static_cast<int32_t>(recvmsg(can_fd, &canMsg, 0));
    // CAN_LOG_INFO << "CanbusMonitor::ReadCan. readBytes: " << readBytes;
    if (readBytes < 0) {
        return RES_CAN_READ_ERR;
    }

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&canMsg);
    if (cmsg != nullptr) {
        tstamp = *(reinterpret_cast<timeval*>(CMSG_DATA(cmsg)));
    }

    return readBytes;
}

int CanbusMonitor::SetCanFiLter(const std::string &canDevice, const struct can_filter& filter) const {
    const auto filterSize = static_cast<socklen_t>(sizeof(filter));
    int32_t ret = -1;
    for(auto sock : sock_list_) {
        if(!strcmp(canDevice.c_str() , sock.can_port.c_str())) {
            ret = setsockopt(sock.can_fd, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, filterSize);
            break;
        }
    }
    if (ret < 0) {
        return RES_SET_SOCKET_OPT_ERR;
    }
    return RES_SUCCESS;
}

int CanbusMonitor::SetCanFilters(const int can_fd, const std::vector<can_filter>& filters) const {
    if (filters.empty()) {
        return RES_ERROR;
    }
    const auto itemSize = static_cast<socklen_t>(sizeof(can_filter));
    const auto filterSize = static_cast<socklen_t>(static_cast<socklen_t>(filters.size()) * itemSize);
    const int32_t ret = setsockopt(can_fd, SOL_CAN_RAW, CAN_RAW_FILTER, filters.data(), filterSize);
    if (ret < 0) {
        return RES_SET_SOCKET_OPT_ERR;
    }
    return RES_SUCCESS;
}

int CanbusMonitor::CloseSocketCan(int sock_can) {
    if (sock_can > 0) {
        (void)close(sock_can);
        sock_can = -1;
    }
    return sock_can;
}

int CanbusMonitor::GetSocketCanfd(const std::string &canDevice) { 
    int ret = -1;
    for(auto sock : sock_list_) {
        if(!strcmp(canDevice.c_str() , sock.can_port.c_str())) {
            ret = sock.can_fd; 
            break;
        }
    }
    return ret;
}

void CanbusMonitor::SetSocketCanNonBlock(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

std::string CanbusMonitor::GetCurrCanDevice(int fd) { 
    std::string canport = NULL;
    for(auto sock : sock_list_) {
        if(sock.can_fd == fd) {
            canport = sock.can_port; 
            break;
        }
    }
    return canport; 
}

void CanbusMonitor::StartCanbusMonitorThread() {
    std::thread monitor_thread = std::thread(&CanbusMonitor::CanDisposeTreadCallback, this);
    monitor_thread.detach();
}

bool CanbusMonitor::IsErrorCan(uint32_t can_id, std::string canPort) {
    if (can_id & CAN_ERR_FLAG) {
        // CAN_LOG_ERROR << "CAN_ERR_FLAG";
        if (can_id & CAN_FRAME_ERROR) {
            // FaultInfo_t faultInfo = GetCommunicationFaultInfo(canPort, CommunicationErrorCase::CAN_ERR_FRAME_OCCUR);
            // // Fault* fault = HzFMAgent::Instance()->GenFault(faultInfo.faultId, faultInfo.faultObj, USE_FM_CHANNEL);
            // // fault->UseTimeBaseDebouncePolicy(3000);
            // // HzFMAgent::Instance()->ReportFaultAsync(fault, 1);
            // hozon::canstack::CanBusReport::Instance().ReportSensorFaultAsync(faultInfo.faultId, faultInfo.faultObj, 1, USE_FM_CHANNEL, 3000); 
        } 
        // else if (can_id & CAN_ERR_BUSOFF) {
        //     is_busoff_ = true;
        //     FaultInfo_t faultInfo = GetCommunicationFaultInfo(can_name_, CommunicationErrorCase::CAN_BUS_OFF);
        //     HzFMAgent::Instance()->ReportFault(HzFMAgent::Instance()->GenFault(faultInfo.faultId, faultInfo.faultObj), 1);
        // }
        std::stringstream ss;
        if (can_id & CAN_ERR_TX_TIMEOUT) {
           ss << "Error Frame receive [" << std::hex << can_id << "] error class: CAN_ERR_TX_TIMEOUT, canPort:" << canPort;
            // TODO: report fault data to FM, FaultID is FAULT_TX_TIMEOUT
        } else if (can_id & CAN_ERR_CRTL) {
            ss << "Error Frame receive [" << std::hex << can_id << "] error class: CAN_ERR_CRTL, canPort:" << canPort;
            // TODO: report fault data to FM, FaultID is FAULT_CRTL_ERR
        } else if (can_id & CAN_ERR_PROT) {
            ss << "Error Frame receive [" << std::hex << can_id  << "] error class: CAN_ERR_PROT, canPort:" << canPort;
            // TODO: report fault data to FM, FaultID is FAULT_PROT_ERR
        } else if (can_id & CAN_ERR_TRX) {
            ss << "Error Frame receive [" << std::hex << can_id  << "] error class: CAN_ERR_TRX, canPort:" << canPort;
            // TODO: report fault data to FM, FaultID is FAULT_TRX_ERR
        } else if (can_id & CAN_ERR_ACK) {
            ss << "Error Frame receive [" << std::hex << can_id  << "] error class: CAN_ERR_ACK, canPort:" << canPort;
            // TODO: report fault data to FM, FaultID is FAULT_ACK_ERR
        } else if (can_id & CAN_ERR_BUSOFF) {
            ss << "Error Frame receive [" << std::hex << can_id  << "] error class: CAN_ERR_BUSOFF, canPort:" << canPort;
            // TODO: report fault data to FM, FaultID is FAULT_BUSOFF_ERR
        } else if (can_id & CAN_ERR_BUSERROR) {
            ss << "Error Frame receive [" << std::hex << can_id  << "] error class: CAN_ERR_BUSERROR, canPort:" << canPort;
            // TODO: report fault data to FM, FaultID is FAULT_BUS_ERR
        }   
        CAN_LOG_ERROR << ss.str();    
        return true;
    }

    return false;
}
void CanbusMonitor::CanDisposeTreadCallback() {
    CAN_LOG_INFO << "can read run..";

    pthread_setname_np(pthread_self(), "CanDisposeCB");
    while (!quit_flag_) {
        canfd_frame receiveFrame;
        struct timeval tstamp;
        std::int32_t read_bytes = 0; 
        fd_set can_fds = can_fds_;
        // CAN_LOG_INFO << "CanFrameMonitor read select maxfd_:" << maxfd_;
        int32_t select_ret = select(maxfd_ + 1, &can_fds, NULL, NULL, NULL);
        if (select_ret == -1 || select_ret == 0) {
            continue;
        }
        if(!quit_flag_) {
            for(auto sock :sock_list_) {
                if((sock.can_fd > 0) && FD_ISSET(sock.can_fd, &can_fds)) {
                    // CAN_LOG_INFO << "CanFrameMonitor read can data " << sock.can_port << "can_fd :" << sock.can_fd;
                    const auto ret = ReadCan(sock.can_fd, receiveFrame, tstamp, read_bytes);
                    // CAN_LOG_INFO << "CanFrameMonitor read can ret " << ret;
                    if (ret == RES_CAN_READ_ERR) {
                        continue;
                    }

                    if (IsErrorCan(receiveFrame.can_id, sock.can_port)) {
                        // CAN_LOG_INFO << "CanFrameMonitor IsErrorCan ";
                        if (ConfigLoader::debug_on_) {
                            std::stringstream ss;
                            
                            ss << "can id:0x" << std::hex << receiveFrame.can_id << " data: " ;
                            for(int index = 0; index < receiveFrame.len; index++) {
                                ss << " 0X" << std::hex << receiveFrame.data[index];
                            }
                            CAN_LOG_DEBUG << ss.str();
                        }
                        continue;
                    } 
                    // else if (is_busoff_) {
                    //     is_busoff_ = false;
                    //     FaultInfo_t faultInfo = GetCommunicationFaultInfo(can_name_, CommunicationErrorCase::CAN_BUS_OFF);
                    //     HzFMAgent::Instance()->ReportFault(HzFMAgent::Instance()->GenFault(faultInfo.faultId, faultInfo.faultObj), 0);  // 故障恢复
                    // }

                    // TODO: Add E2E check.
                    std::uint8_t e2e_ret = E2ESupervision::Instance()->Check(receiveFrame, read_bytes);
                    if (e2e_ret != 0u) {
                        continue;
                    }
                    
                    if (read_bytes == sizeof(can_frame)) {
                        sock.can_parser->ParseCan(*(reinterpret_cast<can_frame*>(&receiveFrame)));
                    } else {
                        sock.can_parser->ParseCanfd(receiveFrame);
                    }     
                }
            }
        }    
    }
    CAN_LOG_INFO << "can read over..";
}

}  // namespace canstack
}
}  // namespace hozon

#include "can_tsync_center/can_tsync.h"  
#include "can_tsync_center/can_tsync_logger.h"
// #include "/usr/local/mdc_sdk_0930/dp_gea/mdc_cross_compiler/sysroot/usr/include/Crc_Api.h"
#include <string.h>  
#include <net/if.h>
#include <linux/can/error.h>
#include <chrono>

namespace hozon {
namespace netaos {
namespace tsync {

CanTsync::CanTsync() {

}

CanTsync::~CanTsync() {

}

int32_t CanTsync::Start(const CanTSyncConfig& config) {
    _config = config;
    int ret = OpenSocket();
    if (ret < 0) {
        return -1;
    }

    _thread = std::make_shared<std::thread>(&CanTsync::TSyncRoutine, this);
    return 0;
}

void CanTsync::Stop() {
    _need_stop = true;
    _thread->join();
    CloseSocket();
}


void CanTsync::CloseSocket() {
    if (_socket > 0) {
        close(_socket);
        _socket = -1;
    }
}

int32_t CanTsync::OpenSocket() {
    CTSC_LOG_INFO_HEAD << "Start to init CAN.";
    _socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (_socket < 0) {
        CTSC_LOG_ERROR_HEAD << "Fail to open socket.";
        CloseSocket();
        return -1;
    }

    ifreq ifr = {};
    char* dst = strncpy(ifr.ifr_name, _config.interface.data(), sizeof(ifr.ifr_name) - 1);
    if (dst != ifr.ifr_name) {
        CTSC_LOG_ERROR_HEAD << "Fail to copy name.";
        CloseSocket();
        return -1;
    }

    int ret = ioctl(_socket, SIOCGIFINDEX, &ifr);
    if (ret < 0) {
        CTSC_LOG_ERROR_HEAD << "Fail to get index on.";
        CloseSocket();
        return -1;
    }

    if ((_config.type == TSyncFrameType::CANFD_16_BYTE) || (_config.type == TSyncFrameType::CANFD_8_BYTE)) {
        CTSC_LOG_INFO_HEAD << "Enable CANFD frames.";
        const int32_t canfd_flag = 1;
        ret = setsockopt(_socket, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &canfd_flag, static_cast<socklen_t>(sizeof(canfd_flag)));
        if (ret < 0) {
            CTSC_LOG_ERROR_HEAD << "Fail to init CANFD.";
            CloseSocket();
            return -1;
        }
    }
    else if (_config.type == TSyncFrameType::CAN_STANDARD) {
        // do nothing
    }
    else {
        CTSC_LOG_ERROR_HEAD << "Unknown type " << static_cast<std::underlying_type<TSyncFrameType>::type>(_config.type);
        return -1;
    }

    can_filter filter;
    filter.can_id = 0x100;
    filter.can_mask = 0xFFFu;
    ret = setsockopt(_socket, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, sizeof(filter));
    if (ret < 0) {
        CTSC_LOG_ERROR_HEAD << "Fail to set filter.";
        CloseSocket();
        return -1;
    }

    const int32_t timestamp_flag = 1;
    ret = setsockopt(_socket, SOL_SOCKET, SO_TIMESTAMPING, &timestamp_flag, static_cast<socklen_t>(sizeof(timestamp_flag)));
    if (ret < 0) {
        CTSC_LOG_ERROR_HEAD << "Fail to set timestamp.";
        CloseSocket();
        return -1;
    }

    // can_err_mask_t err_mask = (CAN_ERR_TX_TIMEOUT | CAN_ERR_CRTL | CAN_ERR_PROT 
    //     | CAN_ERR_TRX | CAN_ERR_ACK | CAN_ERR_BUSOFF | CAN_ERR_BUSERROR);
    // ret = setsockopt(_socket, SOL_CAN_RAW, CAN_RAW_ERR_FILTER, &err_mask, static_cast<socklen_t>(sizeof(err_mask)));
    // if (ret < 0) {
    //     CTSC_LOG_ERROR_HEAD << "Fail to set error filter.";
    //     CloseSocket();
    //     return -1;
    // }

    const int32_t recv_own = 1;
    ret = setsockopt(_socket, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &recv_own, static_cast<socklen_t>(sizeof(recv_own)));
    if (ret < 0) {
        CTSC_LOG_ERROR_HEAD << "Fail to set recv_own.";
        CloseSocket();
        return -1;
    }

    // const int32_t loop_back = 1;
    // ret = setsockopt(_socket, SOL_CAN_RAW, CAN_RAW_LOOPBACK, &loop_back, static_cast<socklen_t>(sizeof(recv_own)));
    // if (ret < 0) {
    //     CTSC_LOG_ERROR_HEAD << "Fail to set loopback.";
    //     CloseSocket();
    //     return -1;
    // }

    sockaddr_can _addr = {};
    _addr.can_family = static_cast<__kernel_sa_family_t>(AF_CAN);
    _addr.can_ifindex = ifr.ifr_ifindex;
    ret = ::bind(_socket, reinterpret_cast<sockaddr*>(&_addr), static_cast<socklen_t>(sizeof(_addr)));
    if (ret < 0) {
        CTSC_LOG_ERROR_HEAD << "Fail to bind.";
        CloseSocket();
        return -1;
    }

    CTSC_LOG_INFO_HEAD << "Succ to init.";
    return 0;
}

struct CanMsg {
    CanMsg(TSyncFrameType type) {
        if (type == TSyncFrameType::CANFD_16_BYTE) {
            _internal_canfd_frame.len = 16;
            data = &_internal_canfd_frame.data[0];
            memset(&_internal_canfd_frame.data[8], 0, 8);
            iov.iov_base = static_cast<void*>(&_internal_canfd_frame);
            iov.iov_len = sizeof(_internal_canfd_frame);
            can_id = &_internal_canfd_frame.can_id;
            max_len = sizeof(canfd_frame);
            _internal_canfd_frame.flags = 0x01;
        }
        else if (type == TSyncFrameType::CANFD_8_BYTE) {
            _internal_canfd_frame.len = 8;
            data = &_internal_canfd_frame.data[0];
            iov.iov_base = static_cast<void*>(&_internal_canfd_frame);
            iov.iov_len = sizeof(_internal_canfd_frame);
            can_id = &_internal_canfd_frame.can_id;
            max_len = sizeof(canfd_frame);
            _internal_canfd_frame.flags = 0x01;
        }
        else if (type == TSyncFrameType::CAN_STANDARD) {
            _internal_can_frame.can_dlc = 8;
            data = &_internal_can_frame.data[0];
            iov.iov_base = static_cast<void*>(&_internal_can_frame);
            iov.iov_len = sizeof(_internal_can_frame);
            can_id = &_internal_can_frame.can_id;
            max_len = sizeof(can_frame);
        }
    }

    canfd_frame _internal_canfd_frame;
    can_frame _internal_can_frame;
    iovec iov;
    char control_buf[CMSG_SPACE(sizeof(timeval))];
    msghdr msg = {
        .msg_name = nullptr,
        .msg_namelen = 0U,
        .msg_iov = &iov,
        .msg_iovlen = 1U,
        .msg_control = control_buf,
        .msg_controllen = sizeof(control_buf),
        .msg_flags = 0,
    };
    uint8_t* data;
    canid_t* can_id;
    int max_len;
};

void CanTsync::TSyncRoutine() {
    uint64_t index = 0;
    uint8_t seq_num = 0;
    int ret = 0;
    timespec sync_time;
    timespec ack_time;

    std::chrono::steady_clock::time_point period_wakeup_timepoint = std::chrono::steady_clock::now();
    while (!_need_stop) {
        ret = SendSync(sync_time, seq_num, index);
        if (ret == 0) {
            ret = RecvSyncAck(ack_time, index);
            if (ret == 0) {
                uint64_t time_diff_ns = (ack_time.tv_sec - sync_time.tv_sec) * 1000 * 1000 * 1000 + ack_time.tv_nsec;
                CTSC_LOG_DEBUG_HEAD << "Received SYNC ack after " << time_diff_ns << "(ns).";
                ret = SendFup(time_diff_ns, seq_num);
                if (ret == 0) {
                    CTSC_LOG_INFO_HEAD << "Sync once, count " << index;
                }
                else {
                    // do nothing
                }
            }
            else {
                // do nothing
            }
        }
        else {
            // do nothing
        }

        ++index;
        ++seq_num;
        std::this_thread::sleep_until(period_wakeup_timepoint + std::chrono::milliseconds(_config.interval_ms * index));
    }
}

int32_t CanTsync::SendSync(timespec& sync_time, uint8_t seq_num, uint64_t index) {
    CanMsg sync_msg(_config.type);
    int nbytes = 0;
    struct cmsghdr* cmsg = nullptr;

    clock_gettime(CLOCK_REALTIME, &sync_time);
    uint32_t lsb_sec = sync_time.tv_sec & 0xFFFFFFFF;
    *sync_msg.can_id = _config.can_id;

    if (_config.crc_enable) {
        sync_msg.data[0] = 0x20; // type
        // sync_msg.data[1] = 0; // CRC
        sync_msg.data[2] = seq_num & 0x0F; // time domain + sequence number 
        sync_msg.data[3] = 0; // user byte0
        sync_msg.data[4] = (lsb_sec >> 24) & 0xFF; // lsb 32bit sec in big endian
        sync_msg.data[5] = (lsb_sec >> 16) & 0xFF;
        sync_msg.data[6] = (lsb_sec >> 8) & 0xFF;
        sync_msg.data[7] = (lsb_sec >> 0) & 0xFF;

        sync_msg.data[1] = CalcCRC(sync_msg.data, seq_num);
    }
    else {
        sync_msg.data[0] = 0x10; // type
        sync_msg.data[1] = 0; // user byte1
        sync_msg.data[2] = seq_num & 0x0F; // time domain + sequence number 
        sync_msg.data[3] = 0; // user byte0
        sync_msg.data[4] = (lsb_sec >> 24) & 0xFF; // lsb 32bit sec in big endian
        sync_msg.data[5] = (lsb_sec >> 16) & 0xFF;
        sync_msg.data[6] = (lsb_sec >> 8) & 0xFF;
        sync_msg.data[7] = (lsb_sec >> 0) & 0xFF;
    }

    cmsg = CMSG_FIRSTHDR(&sync_msg.msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(timeval));
    *(timeval*)CMSG_DATA(cmsg) = (timeval){(int64_t)index, 0};

    nbytes = sendmsg(_socket, &sync_msg.msg, 0);
    if (nbytes != sync_msg.max_len) {
        CTSC_LOG_WARN_HEAD << "Fail to send SYNC frame, ret " << nbytes << ", errno " << errno;
        return -1;
    }
    else {
        CTSC_LOG_DEBUG_HEAD << "Succ to send SYNC.";
        return 0;
    }
}

int32_t CanTsync::RecvSyncAck(timespec& ack_time, uint64_t index) {
    CanMsg ack_msg(_config.type);
    int64_t timeout_ms = _config.timeout_ms;
    // struct cmsghdr* cmsg = nullptr;

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(_socket, &rfds);
    while (timeout_ms > 0) {
        timeval timeout;
        timeout.tv_sec = timeout_ms / 1000;
        timeout.tv_usec = (timeout_ms % 1000) * 1000;

        std::chrono::steady_clock::time_point begin_timepoint = std::chrono::steady_clock::now();
        int ret = select(_socket + 1, &rfds, NULL, NULL, &timeout);
        if (ret == -1) {
            CTSC_LOG_ERROR_HEAD << "Fail to listen on can interface, " << errno;
        } 
        else if (ret == 0) {
            // CTSC_LOG_WARN_HEAD << "Fail to recv loopback SYNC message.";
            // timeout
            break;
        } 
        else {
            CTSC_LOG_DEBUG_HEAD << "Recv something after send SYNC.";
            int nbytes = recvmsg(_socket, &ack_msg.msg, 0);
            if (nbytes < 0) {
                CTSC_LOG_WARN_HEAD << "Fail to recvmsg, " << errno;
            }

            if (*ack_msg.can_id == _config.can_id) {
                if (ack_msg.msg.msg_flags & MSG_CONFIRM) {
                    clock_gettime(CLOCK_REALTIME, &ack_time);
                    std::stringstream ss;
                    ss << "." << std::setw(9) << std::setfill('0') << ack_time.tv_nsec;
                    CTSC_LOG_DEBUG_HEAD << "Received matched SYNC ack, time " << ack_time.tv_sec << ss.str();
                    
                    return 0;
                }
                // CTSC_LOG_DEBUG_HEAD << "Recv ack " << ack_msg.msg.msg_controllen << ", " << ack_msg.msg.msg_flags;
                // for (cmsg = CMSG_FIRSTHDR(&ack_msg.msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&ack_msg.msg, cmsg)) {
                //     if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SO_TIMESTAMP) {
                //         struct timeval *tv = (struct timeval *)CMSG_DATA(cmsg);
                //         std::stringstream ss;
                //         ss << "." << std::setw(6) << std::setfill('0') << tv->tv_usec;
                //         CTSC_LOG_DEBUG_HEAD << "Recv timestamp at " << tv->tv_sec << ss.str();
                //     }
                //     else if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == 100) {
                //         uint64_t recv_index = *(uint64_t*)CMSG_DATA(cmsg);

                //         if (recv_index == index) {
                //             clock_gettime(CLOCK_REALTIME, &ack_time);
                //             CTSC_LOG_DEBUG_HEAD << "Received matched SYNC ack, time " << ack_time.tv_sec << "." << ack_time.tv_nsec;

                //             return 0;
                //         }
                //         else {
                //             CTSC_LOG_DEBUG_HEAD << "Received unmatched SYNC ack, recv index " 
                //                 << recv_index << ", send index " << index;;
                //         }
                //     }
                //     else {
                //         CTSC_LOG_DEBUG_HEAD << "Recv cmsg_type " << cmsg->cmsg_type;
                //     }
                // }
                
            }
            else {
                CTSC_LOG_TRACE_HEAD << "Received unmatched id " << hozon::netaos::log::loghex(*ack_msg.can_id);
            }
        }

        std::chrono::steady_clock::time_point end_timepoint = std::chrono::steady_clock::now();
        timeout_ms -= std::chrono::duration<double, std::milli>(end_timepoint - begin_timepoint).count();

        CTSC_LOG_TRACE_HEAD << "Time remains " << timeout_ms << "(ms).";
    }

    CTSC_LOG_WARN_HEAD << "Fail to receive SYNC ack in " << _config.timeout_ms << "(ms).";
    return -1;
}

int32_t CanTsync::SendFup(uint64_t time_diff_ns, uint8_t seq_num) {
    CanMsg fup_msg(_config.type);
    int nbytes = 0;

    uint32_t nsecs = time_diff_ns & 0xFFFFFFFF;
    if (_config.crc_enable) {
        uint32_t nsecs = time_diff_ns & 0xFFFFFFFF;
        *fup_msg.can_id = _config.can_id;
        fup_msg.data[0] = 0x28; // type
        // fup_msg.data[1] = 0; // CRC
        fup_msg.data[2] = seq_num & 0x0F; // time domain + sequence number 
        fup_msg.data[3] = 0; // SGW & OVS
        fup_msg.data[4] = (nsecs >> 24) & 0xFF; // nano seconds in big endian
        fup_msg.data[5] = (nsecs >> 16) & 0xFF;
        fup_msg.data[6] = (nsecs >> 8) & 0xFF;
        fup_msg.data[7] = (nsecs >> 0) & 0xFF;
    
        fup_msg.data[1] = CalcCRC(fup_msg.data, seq_num);
    }
    else {
        uint32_t nsecs = time_diff_ns & 0xFFFFFFFF;
        *fup_msg.can_id = _config.can_id;
        fup_msg.data[0] = 0x18; // type
        fup_msg.data[1] = 0; // user byte2
        fup_msg.data[2] = seq_num & 0x0F; // time domain + sequence number 
        fup_msg.data[3] = 0; // SGW & OVS
        fup_msg.data[4] = (nsecs >> 24) & 0xFF; // nano seconds in big endian
        fup_msg.data[5] = (nsecs >> 16) & 0xFF;
        fup_msg.data[6] = (nsecs >> 8) & 0xFF;
        fup_msg.data[7] = (nsecs >> 0) & 0xFF;
    }

    nbytes = sendmsg(_socket, &fup_msg.msg, 0);
    if (nbytes != fup_msg.max_len) {
        CTSC_LOG_WARN_HEAD << "Fail to send FUP frame, ret " << nbytes;
        return -1;
    }
    else {
        CTSC_LOG_DEBUG_HEAD << "Succ to send FUP msg, SyncTimeNSec " << nsecs << "(ns).";
        return 0;
    }
}

uint8_t CanTsync::CalcCRC(uint8_t data[], uint8_t seq_num) {
    uint8_t crc_buf[15] = {0};
    int range = 0;

    if (_config.type == TSyncFrameType::CANFD_16_BYTE) {
        range = 14;
    }
    else {
        range = 6;
    }

    for (int i = 0; i < range; ++i) {
        crc_buf[i] = data[i + 2];
    }
    crc_buf[range] = _config.data_id[seq_num];
    
    // return Crc_CalculateCRC8H2F(&crc_buf[0], range + 1, 0xFF, 1);
    return crc_buf[0];
}

}
}
}
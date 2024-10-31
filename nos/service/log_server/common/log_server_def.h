#pragma once

#include <iostream>
#include <array>
#include <vector>

namespace hozon {
namespace netaos {
namespace logserver {

const std::string operation_log_service_name = "tcp://*:15777";
const std::string compress_log_service_name = "tcp://*:15778";

#define OPERATION_LOG_PATH_FOR_MDC_LLVM   ("/opt/usr/")
#define OPERATION_LOG_PATH_FOR_J5         ("/userdata/")
#define OPERATION_LOG_PATH_FOR_ORIN       ("/opt/usr/log/soc_log/")
#define OPERATION_LOG_PATH_DEFAULT        ("/log/")


const std::string mcu_log_ip = "172.16.90.11";
const uint32_t mcu_log_port = 23456;

enum class LogLevel : uint8_t {
    kTrace      = 0x00U,
    kDebug      = 0x01U,
    kInfo       = 0x02U,
    kWarn       = 0x03U,
    kError      = 0x04U,
    kCritical   = 0x05U,
    kOff        = 0x06U
};

enum class OperationLogType : uint8_t {
    tApp        = 0x00U,
    tMw         = 0x01U,
    tBsp        = 0x02U,
    tOs         = 0x03U,
    tSupplier   = 0x04U,
    tOthers     = 0x05U,
    tReserved   = 0x06U
};

struct McuLog {
    struct LogHeader {
        uint8_t app_id;
        uint8_t ctx_id;
        uint8_t level;
        uint8_t seq;
        struct LogTime {
            uint32_t sec;
            uint32_t nsec;
        } stamp;
        uint32_t length;
    } header;

    std::array<uint8_t, 300> log;

    McuLog() {
        header.app_id = 0;
        header.ctx_id = 0;
        header.level = 0;
        header.seq = 0;
        header.stamp.sec = 0;
        header.stamp.nsec = 0;
        header.length = 0;
    }
};

struct LogServerSendFaultInfo {
    uint32_t faultId;
    uint8_t faultObj;
    uint8_t faultStatus;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

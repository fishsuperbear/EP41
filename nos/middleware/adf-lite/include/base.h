#pragma once

#include <cstdint>
#include <memory>
#include "proto/common/header.pb.h"
#include "idl/generated/common.h"
#include "adf/include/data_types/common/latency.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

#define likely(x) __builtin_expect(!!(x), 1) 
#define unlikely(x) __builtin_expect(!!(x), 0)


struct Header {
    uint64_t seq;
    uint64_t timestamp_real_us;
    uint64_t timestamp_virt_us;
    hozon::netaos::adf::LatencyInformation latency_info;
};

struct BaseData {
    Header __header;
    std::shared_ptr<google::protobuf::Message> proto_msg = nullptr;
    std::shared_ptr<IDLBaseType> idl_msg = nullptr;
    virtual void SerializeAsString(std::string &data, std::size_t size) {
        data.assign((char*)this + 8, size - 8);
    }
    virtual void ParseFromString(std::string &data, std::size_t size) {
        if (size - 8 > data.size()) {
            return;
        }
        memcpy((char*)this + 8, data.data(), size - 8);
    }
};

using BaseDataTypePtr = std::shared_ptr<BaseData>;

inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1000 / 1000 / 1000;
}

inline uint64_t TimestampToUs(uint64_t sec, uint64_t nsec) {
    return sec * 1000 * 1000 + nsec / 1000;
}

inline uint64_t TimestampToUs(double d_sec) {
    return (uint64_t)(d_sec * 1000 * 1000);
}

#define DO(statement) \
    if ((statement) < 0) { \
        return -1; \
    }

#define DO_OR_ERROR(statement, errortxt) \
    if ((statement) < 0) { \
        ADF_INTERNAL_LOG_ERROR << errortxt; \
        return -1; \
    }

#define DO_OR_ERROR_EARLY(statement, errortxt) \
    if ((statement) < 0) { \
        ADF_INTERNAL_LOG_ERROR << errortxt; \
        return -1; \
    }

}    
}
}
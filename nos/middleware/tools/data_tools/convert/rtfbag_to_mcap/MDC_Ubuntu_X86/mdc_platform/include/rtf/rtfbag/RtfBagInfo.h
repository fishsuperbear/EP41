/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This file is the implement of class BagInfo.
 *      BagInfo will read a bag file and and provide info for usr
 * Create: 2019-12-23
 * Notes: NA
 */
#ifndef RTF_BAG_INFO_H
#define RTF_BAG_INFO_H

#include <set>

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "rtf/maintaind/impl_type_drivertype.h"

namespace rtf {
namespace rtfbag {
struct BagFileMessageInfo {
    ara::core::String message;
    ara::core::String types;
    uint32_t size;
    double freq;
    rtf::maintaind::DriverType driverType;
};

struct BagFileHeadInfo {
    enum class CompressionType : uint32_t {
        NONE     = 0,    // 未压缩
        GZIP     = 1,    // GZIP
        ZLIB     = 2,    // ZLIB
    };
    ara::core::String path;
    ara::core::String version;
    uint64_t startRecordRealTime;
    uint64_t startRecordVirtualTime;
    uint64_t stopRecordRealTime;
    uint64_t stopRecordVirtualTime;
    uint64_t duration;
    uint64_t start;
    uint64_t end;
    uint64_t size;
    uint32_t messages;
    bool isValid;
    ara::core::Map<ara::core::String, BagFileMessageInfo> messageHead;
    CompressionType compression;
};
class RtfBagInfo {
public:
    RtfBagInfo() = default;
    ~RtfBagInfo() = default;
    static void QueryBagInfo(const std::set<ara::core::String>& paths,
    ara::core::Map<ara::core::String, BagFileHeadInfo>& bagFileInfo);
};
}
}
#endif

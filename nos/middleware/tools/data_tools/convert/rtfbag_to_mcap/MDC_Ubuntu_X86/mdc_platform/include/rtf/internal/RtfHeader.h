/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This file is the implement of class BagFile.
 *      BagFile will create a bag file and provide read or write operation
 * Create: 2019-11-30
 * Notes: NA
 */
#ifndef RTF_HEADER_H
#define RTF_HEADER_H

#include "ara/core/map.h"
#include "ara/core/string.h"

namespace rtf {
namespace rtfbag {
using ReadMap = ara::core::Map<ara::core::String, ara::core::String>;

class RtfHeader {
public:
    RtfHeader() = default;
    ~RtfHeader() = default;

    bool Parse(const uint8_t* buffer, const uint32_t& size);
    ReadMap GetValues() const;

private:
    ReadMap readMap_;
};
}  // namespace rtfbag
}  // namespace rtf
#endif

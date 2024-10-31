/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Header
 */

#ifndef VIZ_HEADER_H
#define VIZ_HEADER_H

#include <cstdint>
#include "viz_times.h"
#include "ara/core/string.h"

namespace mdc {
namespace visual {
using char_t = char;
using float32_t = float;
using float64_t = double;

struct Header {
    uint32_t seq;
    Times stamp;
    ara::core::String frameId;
    Header() : stamp(Times::now()), frameId()
    {
        static uint32_t seqCounter = 0U;
        seqCounter++;
        seq = seqCounter;
    }
    Header(const uint32_t& vSeq, const Times& vStamp, const ara::core::String& vFrameId)
        : seq(vSeq), stamp(vStamp), frameId(vFrameId) {}
};
}
}

#endif // VIZ_HEADER_H

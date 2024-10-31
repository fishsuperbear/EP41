/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: The decalaration of dp adapter operations
 * Create: 2021-01-22
 */
#ifndef RTF_INNER_TYPE_H
#define RTF_INNER_TYPE_H

#include "vrtf/vcc/api/raw_data.h"

namespace rtf {
struct OutMbufData {
    OutMbufData() : privPtr(),
                    mbufPtr()
    {}
    vrtf::core::RawBuffer privPtr;
    vrtf::core::RawBuffer mbufPtr;
};
}

#endif // RTF_INNER_TYPE_H
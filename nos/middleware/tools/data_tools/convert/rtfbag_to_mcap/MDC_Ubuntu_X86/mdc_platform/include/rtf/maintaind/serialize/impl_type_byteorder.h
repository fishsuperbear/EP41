/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_BYTEORDER_H
#define RTF_MAINTAIND_IMPL_TYPE_BYTEORDER_H

#include "rtf/stdtype/impl_type_uint8_t.h"
namespace rtf {
namespace maintaind {
enum class ByteOrder : rtf::stdtype::uint8_t {
    BIGENDIAN,
    LITTLEENDIAN
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_BYTEORDER_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_WIRETYPE_H
#define RTF_MAINTAIND_IMPL_TYPE_WIRETYPE_H

#include "rtf/stdtype/impl_type_uint8_t.h"
namespace rtf {
namespace maintaind {
enum class WireType : rtf::stdtype::uint8_t {
    STATIC,
    DYNAMIC
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_WIRETYPE_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This provide vector for isignalinfo.
 * Create: 2022-03-30
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_ISIGNALINFOVECTOR_H
#define RTF_MAINTAIND_IMPL_TYPE_ISIGNALINFOVECTOR_H
#include "ara/core/vector.h"
#include "rtf/maintaind/pdu/impl_type_isignalinfo.h"
namespace rtf {
namespace maintaind {
using IsignalInfoVector = ara::core::Vector<rtf::maintaind::IsignalInfo>;
} // namespace maintaind
} // namespace rtf

#endif // RTF_MAINTAIND_IMPL_TYPE_ISIGNALINFOVECTOR_H

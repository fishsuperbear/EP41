/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_LISTENERMAP_H
#define RTF_MAINTAIND_IMPL_TYPE_LISTENERMAP_H
#include "ara/core/map.h"
#include "rtf/maintaind/impl_type_listenertype.h"
#include "rtf/maintaind/impl_type_listenerparams.h"

namespace rtf {
namespace maintaind {
using ListenerMap = ara::core::Map<rtf::maintaind::ListenerType, rtf::maintaind::ListenerParams>;
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_LISTENERMAP_H

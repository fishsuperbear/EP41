/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TSYNC_IMPL_TYPE_TSYNCSTATUSFLAG_H
#define ARA_TSYNC_IMPL_TYPE_TSYNCSTATUSFLAG_H

#include "impl_type_uint8.h"
namespace ara {
namespace tsync {
enum class TsyncStatusFlag : UInt8
{
    kTimeOut = 0,
    kSynchronized = 1,
    kSynchToGateway = 2,
    kTimeLeapFuture = 3,
    kTimeLeapPast = 4,
    kHasDLS = 5,
    kDLSActive = 6,
    kSynchronizedNow = 16
};
} // namespace tsync
} // namespace ara


#endif // ARA_TSYNC_IMPL_TYPE_TSYNCSTATUSFLAG_H

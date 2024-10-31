/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HMIDECISIONINFO_H
#define HOZON_HMI_IMPL_TYPE_HMIDECISIONINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/hmi/impl_type_hmilanechangeinfo.h"
#include "hozon/hmi/impl_type_hmiwarnninginfovector.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace hmi {
struct HmiDecisionInfo {
    ::hozon::hmi::HmiLaneChangeInfo LaneChangeInfo;
    ::hozon::hmi::HmiWarnningInfovector WarningInfovector;
    ::uint8_t drive_mode;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(LaneChangeInfo);
        fun(WarningInfovector);
        fun(drive_mode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(LaneChangeInfo);
        fun(WarningInfovector);
        fun(drive_mode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("LaneChangeInfo", LaneChangeInfo);
        fun("WarningInfovector", WarningInfovector);
        fun("drive_mode", drive_mode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("LaneChangeInfo", LaneChangeInfo);
        fun("WarningInfovector", WarningInfovector);
        fun("drive_mode", drive_mode);
    }

    bool operator==(const ::hozon::hmi::HmiDecisionInfo& t) const
    {
        return (LaneChangeInfo == t.LaneChangeInfo) && (WarningInfovector == t.WarningInfovector) && (drive_mode == t.drive_mode);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HMIDECISIONINFO_H

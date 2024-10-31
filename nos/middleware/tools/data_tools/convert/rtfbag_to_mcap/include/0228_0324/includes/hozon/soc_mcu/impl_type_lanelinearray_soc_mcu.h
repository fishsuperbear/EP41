/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_LANELINEARRAY_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_LANELINEARRAY_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_commonheader_soc_mcu.h"
#include "impl_type_uint32.h"
#include "hozon/soc_mcu/impl_type_lanelinedetectionarray_soc_mcu.h"

namespace hozon {
namespace soc_mcu {
struct LaneLineArray_soc_mcu {
    ::hozon::soc_mcu::CommonHeader_soc_mcu header;
    ::UInt32 locSeq;
    bool isLaneChangeToLeft;
    bool isLaneChangeToRight;
    ::hozon::soc_mcu::LaneLineDetectionArray_soc_mcu laneDetectionFrontOut;
    ::hozon::soc_mcu::LaneLineDetectionArray_soc_mcu laneDetectionRearOut;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locSeq);
        fun(isLaneChangeToLeft);
        fun(isLaneChangeToRight);
        fun(laneDetectionFrontOut);
        fun(laneDetectionRearOut);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(isLaneChangeToLeft);
        fun(isLaneChangeToRight);
        fun(laneDetectionFrontOut);
        fun(laneDetectionRearOut);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("isLaneChangeToLeft", isLaneChangeToLeft);
        fun("isLaneChangeToRight", isLaneChangeToRight);
        fun("laneDetectionFrontOut", laneDetectionFrontOut);
        fun("laneDetectionRearOut", laneDetectionRearOut);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("isLaneChangeToLeft", isLaneChangeToLeft);
        fun("isLaneChangeToRight", isLaneChangeToRight);
        fun("laneDetectionFrontOut", laneDetectionFrontOut);
        fun("laneDetectionRearOut", laneDetectionRearOut);
    }

    bool operator==(const ::hozon::soc_mcu::LaneLineArray_soc_mcu& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (isLaneChangeToLeft == t.isLaneChangeToLeft) && (isLaneChangeToRight == t.isLaneChangeToRight) && (laneDetectionFrontOut == t.laneDetectionFrontOut) && (laneDetectionRearOut == t.laneDetectionRearOut);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_LANELINEARRAY_SOC_MCU_H

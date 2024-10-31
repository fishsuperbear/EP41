/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LANELINE_IMPL_TYPE_LANELINEFRAME_H
#define HOZON_LANELINE_IMPL_TYPE_LANELINEFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_boolean.h"
#include "hozon/laneline/impl_type_lanelinearrayarray.h"

namespace hozon {
namespace laneline {
struct LaneLineFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 locSeq;
    ::Boolean isChangeLeft;
    ::Boolean isChangeRight;
    ::hozon::laneline::LaneLineArrayArray laneDetectionFrontOut;
    ::hozon::laneline::LaneLineArrayArray laneDetectionRearOut;

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
        fun(isChangeLeft);
        fun(isChangeRight);
        fun(laneDetectionFrontOut);
        fun(laneDetectionRearOut);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(isChangeLeft);
        fun(isChangeRight);
        fun(laneDetectionFrontOut);
        fun(laneDetectionRearOut);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("isChangeLeft", isChangeLeft);
        fun("isChangeRight", isChangeRight);
        fun("laneDetectionFrontOut", laneDetectionFrontOut);
        fun("laneDetectionRearOut", laneDetectionRearOut);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("isChangeLeft", isChangeLeft);
        fun("isChangeRight", isChangeRight);
        fun("laneDetectionFrontOut", laneDetectionFrontOut);
        fun("laneDetectionRearOut", laneDetectionRearOut);
    }

    bool operator==(const ::hozon::laneline::LaneLineFrame& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (isChangeLeft == t.isChangeLeft) && (isChangeRight == t.isChangeRight) && (laneDetectionFrontOut == t.laneDetectionFrontOut) && (laneDetectionRearOut == t.laneDetectionRearOut);
    }
};
} // namespace laneline
} // namespace hozon


#endif // HOZON_LANELINE_IMPL_TYPE_LANELINEFRAME_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_LANELINEARRAY_H
#define ARA_ADSFI_IMPL_TYPE_LANELINEARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/adsfi/impl_type_laneheader.h"
#include "impl_type_uint8.h"
#include "ara/adsfi/impl_type_lanelinevector.h"

namespace ara {
namespace adsfi {
struct LaneLineArray {
    ::ara::adsfi::LaneHeader header;
    ::UInt8 package_type;
    ::ara::adsfi::LaneLineVector lane_list;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(package_type);
        fun(lane_list);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(package_type);
        fun(lane_list);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("package_type", package_type);
        fun("lane_list", lane_list);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("package_type", package_type);
        fun("lane_list", lane_list);
    }

    bool operator==(const ::ara::adsfi::LaneLineArray& t) const
    {
        return (header == t.header) && (package_type == t.package_type) && (lane_list == t.lane_list);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_LANELINEARRAY_H

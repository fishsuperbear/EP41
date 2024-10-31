/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_TRAJPREDICTARRAY_H
#define ADSFI_IMPL_TYPE_TRAJPREDICTARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "adsfi/impl_type_predictobjectvector.h"
#include "ara/common/impl_type_commontime.h"
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"

namespace adsfi {
struct TrajPredictArray {
    ::ara::common::CommonHeader header;
    ::adsfi::PredictObjectVector object;
    ::ara::common::CommonTime startTime;
    ::ara::common::CommonTime endTime;
    ::UInt8 changeOriginFlag;
    ::UInt8 selfIntent;
    ::UInt16 scenario;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(object);
        fun(startTime);
        fun(endTime);
        fun(changeOriginFlag);
        fun(selfIntent);
        fun(scenario);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(object);
        fun(startTime);
        fun(endTime);
        fun(changeOriginFlag);
        fun(selfIntent);
        fun(scenario);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("object", object);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("changeOriginFlag", changeOriginFlag);
        fun("selfIntent", selfIntent);
        fun("scenario", scenario);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("object", object);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("changeOriginFlag", changeOriginFlag);
        fun("selfIntent", selfIntent);
        fun("scenario", scenario);
    }

    bool operator==(const ::adsfi::TrajPredictArray& t) const
    {
        return (header == t.header) && (object == t.object) && (startTime == t.startTime) && (endTime == t.endTime) && (changeOriginFlag == t.changeOriginFlag) && (selfIntent == t.selfIntent) && (scenario == t.scenario);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_TRAJPREDICTARRAY_H

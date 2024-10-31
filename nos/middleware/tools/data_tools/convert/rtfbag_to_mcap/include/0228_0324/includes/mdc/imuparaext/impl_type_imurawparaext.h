/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_IMUPARAEXT_IMPL_TYPE_IMURAWPARAEXT_H
#define MDC_IMUPARAEXT_IMPL_TYPE_IMURAWPARAEXT_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "ara/gnss/impl_type_geometrypoit.h"
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_uint16.h"
#include "impl_type_uint64.h"
#include "impl_type_double.h"

namespace mdc {
namespace imuparaext {
struct ImuRawParaExt {
    ::ara::gnss::Header header;
    ::ara::gnss::GeometryPoit extrinsic;
    ::UInt8 extValid;
    ::UInt32 extTow;
    ::UInt16 extWeek;
    ::UInt64 extSyncTime;
    ::UInt8 extSyncStatus;
    ::UInt64 reversed1;
    ::UInt64 reversed2;
    ::UInt64 reversed3;
    ::Double reversed4;
    ::Double reversed5;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(extrinsic);
        fun(extValid);
        fun(extTow);
        fun(extWeek);
        fun(extSyncTime);
        fun(extSyncStatus);
        fun(reversed1);
        fun(reversed2);
        fun(reversed3);
        fun(reversed4);
        fun(reversed5);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(extrinsic);
        fun(extValid);
        fun(extTow);
        fun(extWeek);
        fun(extSyncTime);
        fun(extSyncStatus);
        fun(reversed1);
        fun(reversed2);
        fun(reversed3);
        fun(reversed4);
        fun(reversed5);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("extrinsic", extrinsic);
        fun("extValid", extValid);
        fun("extTow", extTow);
        fun("extWeek", extWeek);
        fun("extSyncTime", extSyncTime);
        fun("extSyncStatus", extSyncStatus);
        fun("reversed1", reversed1);
        fun("reversed2", reversed2);
        fun("reversed3", reversed3);
        fun("reversed4", reversed4);
        fun("reversed5", reversed5);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("extrinsic", extrinsic);
        fun("extValid", extValid);
        fun("extTow", extTow);
        fun("extWeek", extWeek);
        fun("extSyncTime", extSyncTime);
        fun("extSyncStatus", extSyncStatus);
        fun("reversed1", reversed1);
        fun("reversed2", reversed2);
        fun("reversed3", reversed3);
        fun("reversed4", reversed4);
        fun("reversed5", reversed5);
    }

    bool operator==(const ::mdc::imuparaext::ImuRawParaExt& t) const
    {
        return (header == t.header) && (extrinsic == t.extrinsic) && (extValid == t.extValid) && (extTow == t.extTow) && (extWeek == t.extWeek) && (extSyncTime == t.extSyncTime) && (extSyncStatus == t.extSyncStatus) && (reversed1 == t.reversed1) && (reversed2 == t.reversed2) && (reversed3 == t.reversed3) && (fabs(static_cast<double>(reversed4 - t.reversed4)) < DBL_EPSILON) && (fabs(static_cast<double>(reversed5 - t.reversed5)) < DBL_EPSILON);
    }
};
} // namespace imuparaext
} // namespace mdc


#endif // MDC_IMUPARAEXT_IMPL_TYPE_IMURAWPARAEXT_H

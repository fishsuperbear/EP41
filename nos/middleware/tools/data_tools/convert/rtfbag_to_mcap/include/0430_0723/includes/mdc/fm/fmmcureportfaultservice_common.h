/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMMCUREPORTFAULTSERVICE_COMMON_H
#define MDC_FM_FMMCUREPORTFAULTSERVICE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/fm/impl_type_fmfaultdata.h"
#include "impl_type_int32.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace fm {
namespace methods {
namespace McuReportFault {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace McuReportFault
} // namespace methods

class FmMcuReportFaultService {
public:
    constexpr FmMcuReportFaultService() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/MdcPlatformServices/PlatformServiceInterface/FmMcuReportFaultServiceInterface/FmMcuReportFaultService");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMMCUREPORTFAULTSERVICE_COMMON_H

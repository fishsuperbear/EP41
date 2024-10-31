/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMREPORTEVENTSERVICE_COMMON_H
#define MDC_FM_FMREPORTEVENTSERVICE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/fm/impl_type_fmfaultdata.h"
#include "impl_type_int32.h"
#include "impl_type_string.h"
#include "impl_type_int32_t.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint32_t.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace fm {
namespace methods {
namespace ReportFault {
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
} // namespace ReportFault
namespace ReportCheckPoint {
struct Output {
    ::int32_t result;

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
} // namespace ReportCheckPoint
namespace ReportProcState {
struct Output {
    ::uint32_t reportPeriod;
    ::int32_t result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(reportPeriod);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(reportPeriod);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (reportPeriod == t.reportPeriod) && (result == t.result);
    }
};
} // namespace ReportProcState
} // namespace methods

class FmReportEventService {
public:
    constexpr FmReportEventService() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/MdcPlatformServices/PlatformServiceInterface/FmServiceInterface/FmReportEventService");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMREPORTEVENTSERVICE_COMMON_H

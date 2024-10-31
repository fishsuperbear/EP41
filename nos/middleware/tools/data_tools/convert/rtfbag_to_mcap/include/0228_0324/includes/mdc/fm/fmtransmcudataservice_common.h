/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMTRANSMCUDATASERVICE_COMMON_H
#define MDC_FM_FMTRANSMCUDATASERVICE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/fm/impl_type_fmmcudataheader.h"
#include "impl_type_uint8vector.h"
#include "mdc/fm/impl_type_fmmcudataretinfo.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace fm {
namespace methods {
namespace ReportMcuData {
struct Output {
    ::mdc::fm::FmMcuDataRetInfo result;

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
} // namespace ReportMcuData
} // namespace methods

class FmTransMcuDataService {
public:
    constexpr FmTransMcuDataService() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/MdcPlatformServices/PlatformServiceInterface/FmServiceInterface/FmTransMcuDataService");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMTRANSMCUDATASERVICE_COMMON_H

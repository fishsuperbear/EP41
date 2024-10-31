/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMFAULTRECEIVESERVICE_COMMON_H
#define MDC_FM_FMFAULTRECEIVESERVICE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/fm/impl_type_hzfaultdata.h"
#include "hozon/fm/impl_type_hzfaultanalysisevent.h"
#include "impl_type_int32.h"
#include "impl_type_stringvector.h"
#include "hozon/fm/impl_type_hzfaultitemvector.h"
#include "hozon/fm/impl_type_hzfaultclustervector.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace fm {
namespace methods {
namespace AlarmReport {
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
} // namespace AlarmReport
namespace GetDataCollectionFile {
struct Output {
    ::StringVector collectFileList;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(collectFileList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(collectFileList);
    }

    bool operator==(const Output& t) const
    {
       return (collectFileList == t.collectFileList);
    }
};
} // namespace GetDataCollectionFile
} // namespace methods

class FmFaultReceiveService {
public:
    constexpr FmFaultReceiveService() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/MdcPlatformServices/PlatformServiceInterface/FmFaultReceiveServiceInterface/FmFaultReceiveService");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMFAULTRECEIVESERVICE_COMMON_H

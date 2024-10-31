/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TSYNC_TSYNCAPSERVICEINTERFACE_COMMON_H
#define ARA_TSYNC_TSYNCAPSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/tsync/impl_type_tsyncstatusflag.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace tsync {
namespace methods {
namespace GetDataPlaneStatusFlag {
struct Output {
    ::ara::tsync::TsyncStatusFlag tsyncStatusFlag;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(tsyncStatusFlag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(tsyncStatusFlag);
    }

    bool operator==(const Output& t) const
    {
       return (tsyncStatusFlag == t.tsyncStatusFlag);
    }
};
} // namespace GetDataPlaneStatusFlag
namespace GetManagePlaneStatusFlag {
struct Output {
    ::ara::tsync::TsyncStatusFlag tsyncStatusFlag;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(tsyncStatusFlag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(tsyncStatusFlag);
    }

    bool operator==(const Output& t) const
    {
       return (tsyncStatusFlag == t.tsyncStatusFlag);
    }
};
} // namespace GetManagePlaneStatusFlag
} // namespace methods

class TsyncApServiceInterface {
public:
    constexpr TsyncApServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/TsyncApServiceInterface/TsyncApServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace tsync
} // namespace ara

#endif // ARA_TSYNC_TSYNCAPSERVICEINTERFACE_COMMON_H

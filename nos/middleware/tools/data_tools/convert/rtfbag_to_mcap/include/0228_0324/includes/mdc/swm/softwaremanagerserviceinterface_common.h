/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_SOFTWAREMANAGERSERVICEINTERFACE_COMMON_H
#define MDC_SWM_SOFTWAREMANAGERSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/swm/impl_type_finggerprinttype.h"
#include "impl_type_int32.h"
#include "impl_type_uint64.h"
#include "mdc/swm/impl_type_gethistoryvectortype.h"
#include "mdc/swm/impl_type_softwarepackagehistoryrecordvector.h"
#include "mdc/swm/impl_type_swclusterinfovectortype.h"
#include "mdc/swm/impl_type_sysversion.h"
#include "mdc/swm/impl_type_response.h"
#include "mdc/swm/impl_type_versionitem.h"
#include "impl_type_string.h"
#include "impl_type_boolean.h"
#include "mdc/swm/impl_type_logvector.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace swm {
namespace methods {
namespace GetFinggerPrint {
struct Output {
    ::mdc::swm::FinggerPrintType fingerPrint;
    ::Int32 ret;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(fingerPrint);
        fun(ret);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(fingerPrint);
        fun(ret);
    }

    bool operator==(const Output& t) const
    {
       return (fingerPrint == t.fingerPrint) && (ret == t.ret);
    }
};
} // namespace GetFinggerPrint
namespace GetHistory {
struct Output {
    ::mdc::swm::GetHistoryVectorType history;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(history);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(history);
    }

    bool operator==(const Output& t) const
    {
       return (history == t.history);
    }
};
} // namespace GetHistory
namespace GetHistoryInfo {
struct Output {
    ::mdc::swm::SoftwarePackageHistoryRecordVector record;
    ::Int32 ret;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(record);
        fun(ret);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(record);
        fun(ret);
    }

    bool operator==(const Output& t) const
    {
       return (record == t.record) && (ret == t.ret);
    }
};
} // namespace GetHistoryInfo
namespace GetSwClusterInfo {
struct Output {
    ::mdc::swm::SwClusterInfoVectorType SwInfo;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SwInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SwInfo);
    }

    bool operator==(const Output& t) const
    {
       return (SwInfo == t.SwInfo);
    }
};
} // namespace GetSwClusterInfo
namespace GetVersionInfo {
struct Output {
    ::mdc::swm::SysVersion SysVersion;
    ::mdc::swm::Response ret;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SysVersion);
        fun(ret);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SysVersion);
        fun(ret);
    }

    bool operator==(const Output& t) const
    {
       return (SysVersion == t.SysVersion) && (ret == t.ret);
    }
};
} // namespace GetVersionInfo
namespace SetFinggerPrint {
struct Output {
    ::Int32 ret;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ret);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ret);
    }

    bool operator==(const Output& t) const
    {
       return (ret == t.ret);
    }
};
} // namespace SetFinggerPrint
namespace GetSpecificVersionInfo {
struct Output {
    ::mdc::swm::VersionItem versionItem;
    ::mdc::swm::Response ret;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(versionItem);
        fun(ret);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(versionItem);
        fun(ret);
    }

    bool operator==(const Output& t) const
    {
       return (versionItem == t.versionItem) && (ret == t.ret);
    }
};
} // namespace GetSpecificVersionInfo
namespace RefreshVersion {
struct Output {
    ::Int32 ret;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ret);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ret);
    }

    bool operator==(const Output& t) const
    {
       return (ret == t.ret);
    }
};
} // namespace RefreshVersion
namespace GetUpdateLogList {
struct Output {
    ::Int32 ret;
    ::mdc::swm::LogVector logList;
    ::String message;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ret);
        fun(logList);
        fun(message);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ret);
        fun(logList);
        fun(message);
    }

    bool operator==(const Output& t) const
    {
       return (ret == t.ret) && (logList == t.logList) && (message == t.message);
    }
};
} // namespace GetUpdateLogList
} // namespace methods

class SoftwareManagerServiceInterface {
public:
    constexpr SoftwareManagerServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/SoftwareManagerServiceInterface/SoftwareManagerServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace swm
} // namespace mdc

#endif // MDC_SWM_SOFTWAREMANAGERSERVICEINTERFACE_COMMON_H

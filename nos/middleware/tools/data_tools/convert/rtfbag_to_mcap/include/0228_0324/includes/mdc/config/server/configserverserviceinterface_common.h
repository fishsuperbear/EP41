/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_COMMON_H
#define MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/config/server/impl_type_paramupdatedata.h"
#include "mdc/config/server/impl_type_servernotifydata.h"
#include "impl_type_string.h"
#include "impl_type_int32_t.h"
#include "impl_type_int32.h"
#include "impl_type_cfgstringlist.h"
#include "impl_type_uint8.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace config {
namespace server {
namespace methods {
namespace AnswerAlive {
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
} // namespace AnswerAlive
namespace DelParam {
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
} // namespace DelParam
namespace GetMonitorClients {
struct Output {
    ::CfgStringList monitorClients;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(monitorClients);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(monitorClients);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (monitorClients == t.monitorClients) && (result == t.result);
    }
};
} // namespace GetMonitorClients
namespace GetParam {
struct Output {
    ::String paramValue;
    ::UInt8 paramType;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(paramValue);
        fun(paramType);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(paramValue);
        fun(paramType);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (paramValue == t.paramValue) && (paramType == t.paramType) && (result == t.result);
    }
};
} // namespace GetParam
namespace MonitorParam {
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
} // namespace MonitorParam
namespace SetParam {
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
} // namespace SetParam
namespace UnMonitorParam {
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
} // namespace UnMonitorParam
namespace InitClient {
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
} // namespace InitClient
} // namespace methods

class ConfigServerServiceInterface {
public:
    constexpr ConfigServerServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/MdcPlatformServices/ConfigServerServiceInterface/ConfigServerServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace server
} // namespace config
} // namespace mdc

#endif // MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_COMMON_H

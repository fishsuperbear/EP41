/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_COMMON_H
#define HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/config/server/impl_type_hzparamupdatedata.h"
#include "hozon/config/server/impl_type_hzservernotifydata.h"
#include "hozon/config/server/impl_type_struct_config_array.h"
#include "impl_type_string.h"
#include "impl_type_int32_t.h"
#include "impl_type_int32.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace config {
namespace server {
namespace methods {
namespace HzAnswerAlive {
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
} // namespace HzAnswerAlive
namespace HzDelParam {
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
} // namespace HzDelParam
namespace HzGetMonitorClients {
struct Output {
    ::UInt32 monitorcount;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(monitorcount);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(monitorcount);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (monitorcount == t.monitorcount) && (result == t.result);
    }
};
} // namespace HzGetMonitorClients
namespace HzGetParam {
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
} // namespace HzGetParam
namespace HzMonitorParam {
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
} // namespace HzMonitorParam
namespace HzSetParam {
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
} // namespace HzSetParam
namespace HzUnMonitorParam {
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
} // namespace HzUnMonitorParam
namespace VehicleCfgUpdateResFromMcu {
struct Output {
    ::UInt8 Result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Result);
    }

    bool operator==(const Output& t) const
    {
       return (Result == t.Result);
    }
};
} // namespace VehicleCfgUpdateResFromMcu
namespace HzGetVehicleCfgParam {
struct Output {
    ::UInt8 paramValue;
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(paramValue);
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(paramValue);
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (paramValue == t.paramValue) && (returnCode == t.returnCode);
    }
};
} // namespace HzGetVehicleCfgParam
namespace HzGetVINCfgParam {
struct Output {
    ::String paramValue;
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(paramValue);
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(paramValue);
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (paramValue == t.paramValue) && (returnCode == t.returnCode);
    }
};
} // namespace HzGetVINCfgParam
} // namespace methods

class HzConfigServerSomeipServiceInterface {
public:
    constexpr HzConfigServerSomeipServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/HzConfigServerServiceInterface/HzConfigServerSomeipServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace server
} // namespace config
} // namespace hozon

#endif // HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_COMMON_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_COMMON_H
#define HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/config/server/impl_type_struct_config_array.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace config {
namespace server {
namespace methods {
namespace ReadVehicleConfig {
struct Output {
    ::hozon::config::server::struct_config_array vehicleConfigInfo;
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vehicleConfigInfo);
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vehicleConfigInfo);
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (vehicleConfigInfo == t.vehicleConfigInfo) && (returnCode == t.returnCode);
    }
};
} // namespace ReadVehicleConfig
namespace WriteVehicleConfig {
struct Output {
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (returnCode == t.returnCode);
    }
};
} // namespace WriteVehicleConfig
namespace ReadVINConfig {
struct Output {
    ::String VIN;
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(VIN);
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(VIN);
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (VIN == t.VIN) && (returnCode == t.returnCode);
    }
};
} // namespace ReadVINConfig
namespace WriteVINConfig {
struct Output {
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (returnCode == t.returnCode);
    }
};
} // namespace WriteVINConfig
namespace ReadSNConfig {
struct Output {
    ::String SN;
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SN);
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SN);
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (SN == t.SN) && (returnCode == t.returnCode);
    }
};
} // namespace ReadSNConfig
namespace WriteSNConfig {
struct Output {
    ::UInt8 returnCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(returnCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(returnCode);
    }

    bool operator==(const Output& t) const
    {
       return (returnCode == t.returnCode);
    }
};
} // namespace WriteSNConfig
} // namespace methods

class HzConfigServerDdsServiceInterface {
public:
    constexpr HzConfigServerDdsServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/HzConfigServerServiceInterface/HzConfigServerDdsServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace server
} // namespace config
} // namespace hozon

#endif // HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_COMMON_H

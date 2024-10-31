/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_COMMON_H
#define HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/tsp_pki/impl_type_hduuidresult.h"
#include "hozon/tsp_pki/impl_type_uploadtokenresult.h"
#include "hozon/tsp_pki/impl_type_remoteconfigresult.h"
#include "hozon/tsp_pki/impl_type_pkistatus.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace tsp_pki {
namespace methods {
namespace RequestHdUuid {
struct Output {
    ::hozon::tsp_pki::HdUuidResult HdUuidResult;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(HdUuidResult);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(HdUuidResult);
    }

    bool operator==(const Output& t) const
    {
       return (HdUuidResult == t.HdUuidResult);
    }
};
} // namespace RequestHdUuid
namespace RequestUploadToken {
struct Output {
    ::hozon::tsp_pki::UploadTokenResult UploadTokenResult;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(UploadTokenResult);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(UploadTokenResult);
    }

    bool operator==(const Output& t) const
    {
       return (UploadTokenResult == t.UploadTokenResult);
    }
};
} // namespace RequestUploadToken
namespace RequestRemoteConfig {
struct Output {
    ::hozon::tsp_pki::RemoteConfigResult RemoteConfigResult;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(RemoteConfigResult);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(RemoteConfigResult);
    }

    bool operator==(const Output& t) const
    {
       return (RemoteConfigResult == t.RemoteConfigResult);
    }
};
} // namespace RequestRemoteConfig
namespace ReadPkiStatus {
struct Output {
    ::hozon::tsp_pki::Pkistatus Pkistatus;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Pkistatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Pkistatus);
    }

    bool operator==(const Output& t) const
    {
       return (Pkistatus == t.Pkistatus);
    }
};
} // namespace ReadPkiStatus
} // namespace methods

class HozonInterface_TspPki {
public:
    constexpr HozonInterface_TspPki() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_TspPki");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace tsp_pki
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_COMMON_H

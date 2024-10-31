/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_VO_VOCONFIGSERVICEINTERFACE_COMMON_H
#define MDC_VO_VOCONFIGSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/vo/impl_type_chninfolist.h"
#include "impl_type_uint8_t.h"
#include "mdc/vo/impl_type_chnattr.h"
#include "impl_type_int32_t.h"
#include "mdc/vo/impl_type_voenum.h"
#include "mdc/vo/impl_type_displayattrenum.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace vo {
namespace methods {
namespace GetVoConfig {
struct Output {
    ::mdc::vo::ChnInfoList chnInfoList;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(chnInfoList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(chnInfoList);
    }

    bool operator==(const Output& t) const
    {
       return (chnInfoList == t.chnInfoList);
    }
};
} // namespace GetVoConfig
namespace GetVoChnAttr {
struct Output {
    ::mdc::vo::ChnAttr chnAttr;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(chnAttr);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(chnAttr);
    }

    bool operator==(const Output& t) const
    {
       return (chnAttr == t.chnAttr);
    }
};
} // namespace GetVoChnAttr
namespace SetVoChnAttr {
struct Output {
    ::int32_t ret;

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
} // namespace SetVoChnAttr
namespace SetVoSource {
struct Output {
    ::int32_t ret;

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
} // namespace SetVoSource
namespace GetVoSource {
struct Output {
    ::mdc::vo::VoEnum videoSource;
    ::int32_t ret;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(videoSource);
        fun(ret);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(videoSource);
        fun(ret);
    }

    bool operator==(const Output& t) const
    {
       return (videoSource == t.videoSource) && (ret == t.ret);
    }
};
} // namespace GetVoSource
namespace SetChnDisplayAttr {
struct Output {
    ::int32_t ret;

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
} // namespace SetChnDisplayAttr
} // namespace methods

class VoConfigServiceInterface {
public:
    constexpr VoConfigServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/VoConfigServiceInterface/VoConfigServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace vo
} // namespace mdc

#endif // MDC_VO_VOCONFIGSERVICEINTERFACE_COMMON_H

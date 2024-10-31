/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HM_HZHMSERVICEINTERFACE_COMMON_H
#define HOZON_HM_HZHMSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_string.h"
#include "impl_type_uint32_t.h"
#include "impl_type_uint8_t.h"
#include "hozon/hm/impl_type_transition.h"
#include "hozon/hm/impl_type_transitions.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hm {
namespace methods {
namespace RegistAliveTask {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace RegistAliveTask
namespace UnRegistAliveTask {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace UnRegistAliveTask
namespace RegistDeadlineTask {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace RegistDeadlineTask
namespace UnRegistDeadlineTask {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace UnRegistDeadlineTask
namespace RegistLogicTask {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace RegistLogicTask
namespace ReportLogicCheckpoint {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace ReportLogicCheckpoint
namespace UnRegistLogicTask {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace UnRegistLogicTask
namespace ReportProcAliveCheckpoint {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace ReportProcAliveCheckpoint
} // namespace methods

class HzHmServiceInterface {
public:
    constexpr HzHmServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HzHmServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hm
} // namespace hozon

#endif // HOZON_HM_HZHMSERVICEINTERFACE_COMMON_H

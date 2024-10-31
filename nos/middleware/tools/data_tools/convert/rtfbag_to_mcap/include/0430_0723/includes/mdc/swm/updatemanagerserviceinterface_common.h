/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_UPDATEMANAGERSERVICEINTERFACE_COMMON_H
#define MDC_SWM_UPDATEMANAGERSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/swm/impl_type_swnamevectortype.h"
#include "mdc/swm/impl_type_response.h"
#include "impl_type_uint8.h"
#include "impl_type_int32.h"
#include "impl_type_string.h"
#include "mdc/swm/impl_type_transferidtype.h"
#include "mdc/swm/impl_type_precheckresult.h"
#include "mdc/swm/impl_type_itemstatevector.h"
#include "mdc/swm/impl_type_packagemanagerstatustype.h"
#include "impl_type_int8.h"
#include "mdc/swm/impl_type_verifycheckresultvectortype.h"
#include "impl_type_boolean.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace swm {
namespace methods {
namespace Activate {
struct Output {
    ::mdc::swm::Response ret;

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
} // namespace Activate
namespace Finish {
struct Output {
    ::mdc::swm::Response ret;

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
} // namespace Finish
namespace GetActivationProgress {
struct Output {
    ::UInt8 process;
    ::Int32 errcode;
    ::String message;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(process);
        fun(errcode);
        fun(message);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(process);
        fun(errcode);
        fun(message);
    }

    bool operator==(const Output& t) const
    {
       return (process == t.process) && (errcode == t.errcode) && (message == t.message);
    }
};
} // namespace GetActivationProgress
namespace GetSwProcessProgress {
struct Output {
    ::UInt8 progress;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(progress);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(progress);
    }

    bool operator==(const Output& t) const
    {
       return (progress == t.progress);
    }
};
} // namespace GetSwProcessProgress
namespace GetUpdatePreCheckProgress {
struct Output {
    ::UInt8 process;
    ::Int32 errcode;
    ::String message;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(process);
        fun(errcode);
        fun(message);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(process);
        fun(errcode);
        fun(message);
    }

    bool operator==(const Output& t) const
    {
       return (process == t.process) && (errcode == t.errcode) && (message == t.message);
    }
};
} // namespace GetUpdatePreCheckProgress
namespace GetUpdatePreCheckResult {
struct Output {
    ::mdc::swm::PreCheckResult ret;

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
} // namespace GetUpdatePreCheckResult
namespace GetUpdateProgress {
struct Output {
    ::UInt8 process;
    ::Int32 errcode;
    ::String message;
    ::mdc::swm::ItemStateVector subItems;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(process);
        fun(errcode);
        fun(message);
        fun(subItems);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(process);
        fun(errcode);
        fun(message);
        fun(subItems);
    }

    bool operator==(const Output& t) const
    {
       return (process == t.process) && (errcode == t.errcode) && (message == t.message) && (subItems == t.subItems);
    }
};
} // namespace GetUpdateProgress
namespace GetUpdateStatus {
struct Output {
    ::mdc::swm::PackageManagerStatusType Status;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Status);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Status);
    }

    bool operator==(const Output& t) const
    {
       return (Status == t.Status);
    }
};
} // namespace GetUpdateStatus
namespace ProcessSwPackage {
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
} // namespace ProcessSwPackage
namespace Update {
struct Output {
    ::mdc::swm::Response ret;

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
} // namespace Update
namespace UpdatePreCheck {
struct Output {
    ::mdc::swm::Response ret;

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
} // namespace UpdatePreCheck
namespace GetVerifyList {
struct Output {
    ::mdc::swm::VerifyCheckResultVectorType checkResult;
    ::Boolean needArchive;
    ::Int8 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(checkResult);
        fun(needArchive);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(checkResult);
        fun(needArchive);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (checkResult == t.checkResult) && (needArchive == t.needArchive) && (result == t.result);
    }
};
} // namespace GetVerifyList
namespace Rollback {
struct Output {
    ::mdc::swm::Response ret;

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
} // namespace Rollback
namespace ActivateByMode {
struct Output {
    ::mdc::swm::Response ret;

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
} // namespace ActivateByMode
} // namespace methods

class UpdateManagerServiceInterface {
public:
    constexpr UpdateManagerServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/UpdateManagerServiceInterface/UpdateManagerServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace swm
} // namespace mdc

#endif // MDC_SWM_UPDATEMANAGERSERVICEINTERFACE_COMMON_H

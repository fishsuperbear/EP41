/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DISP_DISPLAY_DISPLAYIMAGEMBUFSERVICEINTERFACE_COMMON_H
#define MDC_DISP_DISPLAY_DISPLAYIMAGEMBUFSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/display/impl_type_displayimagembufstruct.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace disp {
namespace display {

class DisplayImageMbufServiceInterface {
public:
    constexpr DisplayImageMbufServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/DisplayServiceInterfacePkg/DisplayImageMbufServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace display
} // namespace disp
} // namespace mdc

#endif // MDC_DISP_DISPLAY_DISPLAYIMAGEMBUFSERVICEINTERFACE_COMMON_H

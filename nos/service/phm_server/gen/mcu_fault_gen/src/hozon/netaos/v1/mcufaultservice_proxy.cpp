/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file mcufaultservice_proxy.cpp
 * @brief proxy.cpp
 *
 */


#include "hozon/netaos/v1/mcufaultservice_proxy.h"
#include "ara/com/internal/proxy.h"
#include "ara/com/internal/manifest_config.h"
STRUCTURE_REFLECTION_DEF(hozon::netaos::v1::proxy::methods::McuFaultService::FaultReport::Output,FaultReportResult);
extern ara::com::runtime::ComServiceManifestConfig mcufaultservice_1_0_manifest_config;

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace proxy{
static const ara::com::SomeipTransformationProps prop {
ara::com::DataAlignment::DataAlignment_8,                // alignment
ara::com::ByteOrderEnum::BYTE_ORDER_BIG_ENDIAN,                 // byteOrder
false,                  // implementsLegacyString
false,                  // isDynamicLengthFieldSize
true,                 // isSessionHandlingActive
ara::com::LengthFieldSize::LengthFieldSize_16,                   // sizeOfArrayLengthField
ara::com::LengthFieldSize::LengthFieldSize_8,                  // sizeOfStringLengthField
ara::com::LengthFieldSize::LengthFieldSize_32,                 // sizeOfStructLengthField
ara::com::LengthFieldSize::LengthFieldSize_32,                   // sizeOfUnionLengthField
ara::com::TypeSelectorFieldSize::TypeSelectorFieldSize_32,      // sizeOfUnionTypeSelectorField
ara::com::StringEncoding::UTF8     // stringEncoding
};

static const ara::com::SomeipTransformationProps kDefaultSomeipTransformationProps {
    ara::com::DataAlignment::DataAlignment_8,                  // alignment
    ara::com::ByteOrderEnum::BYTE_ORDER_BIG_ENDIAN,            // byteOrder
    false,                                                     // implementsLegacyString
    false,                                                     // isDynamicLengthFieldSize
    false,                                                     // isSessionHandlingActive
    ara::com::LengthFieldSize::LengthFieldSize_32,             // sizeOfArrayLengthField
    ara::com::LengthFieldSize::LengthFieldSize_32,             // sizeOfStringLengthField
    ara::com::LengthFieldSize::LengthFieldSize_0,             // sizeOfStructLengthField
    ara::com::LengthFieldSize::LengthFieldSize_32,             // sizeOfUnionLengthField
    ara::com::TypeSelectorFieldSize::TypeSelectorFieldSize_32,  // sizeOfUnionTypeSelectorField
    ara::com::StringEncoding::UTF8                             // stringEncoding
};
namespace methods{
namespace McuFaultService{
FaultReport::FaultReport(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{
    setMethodResponseParameters<FaultReport::Output>(instance_, idx_,prop);
}
ara::core::Future<FaultReport::Output>FaultReport::operator()(const ::hozon::netaos::FaultDataStruct& FaultData)
{
    return sendMethodAsyncRequest<FaultReport::Output>(instance_, idx_,prop , FaultData);
}
FaultToHMI::FaultToHMI(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{
}
void FaultToHMI::operator()(const ::hozon::netaos::signal& FaultToHmi)
{
    sendMethodAsyncRequest(instance_, idx_,prop , FaultToHmi);
}
} // namespace McuFaultService
} // namespace methods


McuFaultServiceProxy::McuFaultServiceProxy(const ara::com::HandleType& handle_type)
: instance_(std::make_shared<ara::com::runtime::ProxyInstance>("mcufaultservice_1_0"))
, FaultReport("FaultReport", instance_, instance_->methodIdx("FaultReport"))
, FaultToHMI("FaultToHMI", instance_, instance_->methodIdx("FaultToHMI"))
{
    instance_->updateHandleType(handle_type);
}


ara::com::ServiceHandleContainer<ara::com::HandleType> McuFaultServiceProxy::FindService(ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcufaultservice_1_0", instance);
}

ara::com::ServiceHandleContainer<ara::com::HandleType> McuFaultServiceProxy::FindService(ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcufaultservice_1_0", instance);
}

ara::com::FindServiceHandle McuFaultServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcufaultservice_1_0", instance);
}

ara::com::FindServiceHandle McuFaultServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcufaultservice_1_0", instance);
}

void McuFaultServiceProxy::StopFindService(ara::com::FindServiceHandle handle)
{
    ara::com::runtime::ProxyInstance::StopFindService(handle, "mcufaultservice_1_0");
}

ara::com::HandleType McuFaultServiceProxy::GetHandle() const
{
    return instance_->getHandle();;
}

static ara::core::Result<void> McuFaultServiceProxyInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("mcufaultservice_1_0", &mcufaultservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(McuFaultServiceProxyInitialize);
}
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */
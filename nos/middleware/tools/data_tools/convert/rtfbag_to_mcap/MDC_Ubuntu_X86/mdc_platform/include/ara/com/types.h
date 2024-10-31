/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: This file provides an interface related to communication management.
 * Create: 2019-07-01
 */
#ifndef ARA_COM_TYPES_H
#define ARA_COM_TYPES_H

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include "ara/com/internal/bindindex.h"
#include "vrtf/vcc/api/types.h"
#include "ara/core/string_view.h"
#include "ara/core/instance_specifier.h"
#include "vrtf/vcc/api/raw_data.h"
#include "vrtf/vcc/api/recv_buffer.h"
#include "vrtf/vcc/utils/plog_info.h"
#include "vrtf/vcc/api/subscriber_listener.h"
namespace ara {
namespace com {
namespace e2e {
using ProfileCheckStatus = vrtf::com::e2exf::ProfileCheckStatus;
using SMState = vrtf::com::e2exf::SMState;
using Result = vrtf::com::e2exf::Result;
} /* End e2e namespace */
template<class T>
using SamplePtr = vrtf::vcc::api::types::SamplePtr<T>;
using RawMemory = vrtf::vcc::api::types::RawBuffer;
using RecvMemory = vrtf::vcc::api::types::RecvBuffer;
template<typename T>
struct IsRawMemory {
    static const bool value = false;
};

template<>
struct IsRawMemory<RawMemory> {
    static const bool value = true;
};
// SWS_CM_01010
class ServiceIdentifierType {
public:
    ServiceIdentifierType() = delete;
    ServiceIdentifierType(const ServiceIdentifierType& other) = default;
    ~ServiceIdentifierType(void) = default;
    bool operator==(const ServiceIdentifierType& other) const
    {
        return serviceName_ == other.GetServiceName();
    }

    bool operator<(const ServiceIdentifierType& other) const
    {
        return serviceName_ < other.GetServiceName();
    }
    ServiceIdentifierType& operator=(const ServiceIdentifierType& other) = default;
    //  Complies with 1803 specifications.
    ara::core::StringView toString() const
    {
        return ara::core::StringView(serviceName_);
    }
    // Prohibit to use internal interface by Application!!!!
    explicit constexpr ServiceIdentifierType(const char* serviceName)
    {
        serviceName_ = serviceName;
    }
    // Prohibit to use internal interface by Application!!!!
    internal::ServiceNameType GetServiceName() const
    {
        return std::string(serviceName_);
    }
private:
    const char* serviceName_ = internal::UNDEFINED_SERVICE_NAME;
};

// SWS_CM_01010
class ServiceVersionType {
public:
    ServiceVersionType() = delete;
    ServiceVersionType(const ServiceVersionType& other) = default;
    bool operator==(const ServiceVersionType& other) const
    {
        return versionView_ == other.GetVersion();
    }
    bool operator<(const ServiceVersionType& other) const
    {
        return versionView_ < other.GetVersion();
    }
    ServiceVersionType& operator=(const ServiceVersionType& other) = default;
    ara::core::StringView toString() const
    {
        return ara::core::StringView(versionView_);
    }

    // Prohibit to use internal interface by Application!!!!
    explicit constexpr ServiceVersionType(const char* version)
        : versionView_(version) {}

    // Prohibit to use internal interface by Application!!!!
    const char* GetVersion() const
    {
        return versionView_;
    }
private:
    const char* versionView_;
};

using InstanceIdentifier = vrtf::vcc::api::types::InstanceIdentifier;

using InstanceIdentifierContainer = std::vector<InstanceIdentifier>;

// 18-10 SWS_CM_00307
template <typename T>
using SampleContainer = vrtf::vcc::api::types::SampleContainer<T>;

// 18-10 SWS_CM_00304
template <typename T>
using ServiceHandleContainer = vrtf::vcc::api::types::ServiceHandleContainer<T>;

// 18-10 SWS_CM_00308
template <typename T>
using SampleAllocateePtr = std::unique_ptr<T>;

// 18-10 SWS_CM_00309
using EventReceiveHandler = vrtf::vcc::api::types::EventReceiveHandler;

// 18-10 SWS_CM_00303
using FindServiceHandle = vrtf::vcc::api::types::FindServiceHandle;

// 18-10 SWS_CM_00383
template <typename T>
using FindServiceHandler = vrtf::vcc::api::types::FindServiceHandler<T>;

using HandleType = vrtf::vcc::api::types::HandleType;

using DriverType = vrtf::vcc::api::types::DriverType;

// 18-10 SWS_CM_00310
using SubscriptionState = vrtf::vcc::api::types::EventSubscriptionState;

// 18-10 SWS_CM_00311
using SubscriptionStateChangeHandler = std::function<void(SubscriptionState)>;

// 18-10 SWS_CM_00301
// Method Related Data Types
using MethodCallProcessingMode = vrtf::vcc::api::types::MethodCallProcessingMode;

using SerializationType = vrtf::serialize::SerializationType;
namespace internal {
    using PlogServerTimeStampNode = vrtf::vcc::utils::PlogServerTimeStampNode;
    using PlogInfo = vrtf::vcc::utils::PlogInfo;
    using SampleInfoImpl = vrtf::vcc::api::types::internal::SampleInfoImpl;
    using PlogDriverType = vrtf::vcc::utils::PlogDriverType;
}

enum class CRCVerificationType: uint8_t {
    NOT = 0,
    WEAK,
    STRONG
};

using ListenerStatus = vrtf::vcc::api::types::ListenerStatus;
using ListenerMask = vrtf::vcc::api::types::ListenerMask;
using SubscriberListener = vrtf::vcc::api::types::SubscriberListener;
using ListenerTimeoutInfo = vrtf::vcc::api::types::ListenerTimeoutInfo;
}
}

#endif

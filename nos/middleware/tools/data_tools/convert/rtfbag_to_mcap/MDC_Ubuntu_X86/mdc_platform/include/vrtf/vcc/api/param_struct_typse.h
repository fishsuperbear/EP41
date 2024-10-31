/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Define Info types in communication mannger
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_API_INTERNAL_INFOTYPE_H
#define VRTF_VCC_API_INTERNAL_INFOTYPE_H
#include <cstdint>
#include <vector>
#include <deque>
#include <functional>
#include <memory>
#include <map>
#include <sstream>
#include "ara/core/string.h"
#include "ara/core/future.h"
#include "ara/core/promise.h"
#include "ara/core/vector.h"
#include "vrtf/vcc/api/sample_ptr.h"
#include "vrtf/vcc/utils/plog_info.h"
#include "vrtf/vcc/serialize/serialize_config.h"
// tools
#include "vrtf/vcc/api/raw_data.h"

/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
#ifndef COVERITY_TAINT_SET_DEFINITION
#define COVERITY_TAINT_SET_DEFINITION
/**
 * @brief Function for Stain Modeling
 * @details The function is used only when the compilation macro AOS_TAINT is enabled.
 */
static void Coverity_Tainted_Set(void *buf){}
#endif
#endif
namespace vrtf {
namespace core {
template <class T>
using Future = ara::core::Future<T>;
template <class T>
using Promise = ara::core::Promise<T>;
using ErrorCode = ara::core::ErrorCode;
using ErrorDomain = ara::core::ErrorDomain;
using String = ara::core::String;
using Exception = ara::core::Exception;
using FutureException = ara::core::FutureException;
using FutureErrorDomain = ara::core::FutureErrorDomain;
template <class T, class E = vrtf::core::ErrorCode>
using Result = ara::core::Result<T, E>;
template <class T>
using Vector = ara::core::Vector<T>;
}
namespace vcc {
namespace utils {
class ThreadPool;
}
namespace api {
namespace types {
namespace internal {
enum class MethodType : uint8_t {
    GENERAL_METHOD,
    FIELD_SETTER,
    FIELD_GETTER
};
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
#ifndef COVERITY_TAINT_SET_DEFINITION
#define COVERITY_TAINT_SET_DEFINITION
    static void Coverity_Tainted_Set(void* config) {}
#endif
#endif

class ErrorDomainInfo {
public:
    ErrorDomainInfo(const std::string& domainName,
                    const vrtf::core::ErrorDomain::IdType& domainId)
        : name_(domainName), id_(domainId)
    {
    }
    ~ErrorDomainInfo() = default;
    vrtf::core::ErrorDomain::IdType Id() const
    {
        return id_;
    }

    std::string Name() const
    {
        return name_;
    }

private:
    std::string name_;
    vrtf::core::ErrorDomain::IdType id_;
};
}
enum class ThreadMode : uint8_t {
    EVENT  = 0x00U,
    POLL   = 0x01U,
};
enum class DriverType: uint8_t {
    PROLOCTYPE,
    DDSTYPE,
    SOMEIPTYPE,
    INVALIDTYPE
};
enum class EntityType: uint8_t {
    EVENT = 0U,
    METHOD = 1U,
    UNKNOW = 255U
};
enum class MOVEBIT : std::uint8_t {
    MOVE8BIT = 8U,
    MOVE16BIT = 16U,
    MOVE32BIT = 32U,
    MOVE48BIT = 48U,
    MOVE56BIT = 56U
};
using SerializationType = vrtf::serialize::SerializationType;
using StructSerializationPolicy = vrtf::serialize::StructSerializationPolicy;
using SerializeType = vrtf::serialize::SerializeType;
using SerializeConfig = vrtf::serialize::SerializeConfig;

const std::map<DriverType, std::string> DRIVER_TYPE_MAP {
    std::map<DriverType, std::string>::value_type(DriverType::PROLOCTYPE, "PROLOC"),
    std::map<DriverType, std::string>::value_type(DriverType::DDSTYPE, "DDS"),
    std::map<DriverType, std::string>::value_type(DriverType::SOMEIPTYPE, "SOMEIP")
};

enum class MethodType : uint8_t {
    REQUEST,
    REPLY
};

enum class ReceiveType: uint8_t {
    OK,
    FULLSLOT,
    EMPTY_CONTAINER,
    E2EFAIL
};

enum class VersionDrivenFindBehavior : uint8_t {
    EXACT_OR_ANY_MINOR_VERSION = 0u,
    MINIMUM_MINOR_VERSION
};

enum class AppState : uint8_t {
    APP_REGISTERED = 0x0U,
    APP_DEREGISTERED = 0x1U
};

enum class CacheStatus : uint8_t {
    EMPTY   = 0x00U,
    NORMAL  = 0x01U,
    FULL    = 0x02U,
    UNKNOWN = 0x03U
};

enum class StatisticKind : uint8_t {
    RECV_PACKS            = 0x00U,
    DISCARD_PACKS         = 0x01U,
    READ_BY_USER          = 0x02U,
    SEND_PACKS            = 0x03U,
    DISCARD_BY_SENDER     = 0x04U,
    LATENCY_AVG           = 0x05U,
    LATENCY_MAX           = 0x06U,
    LATENCY_MAX_TIMESTAMP = 0x07U
};

enum class ThreadPoolType : uint8_t {
    RTF_COM = 0x00U,
    ARA_COM = 0x01U
};

using ThreadPoolPair = std::pair<ThreadPoolType, std::shared_ptr<vrtf::vcc::utils::ThreadPool>>;

using ServiceId = std::uint16_t;
using ServiceName = std::string;
using InstanceId = std::string;
using NetworkIp = std::pair<std::string, bool>;
using EntityId = std::uint32_t;
using ErrorCode = std::int32_t;
using MajorVersionId = std::uint8_t;
using MinorVersionId = std::uint32_t;
using ClientId = std::uint16_t;
using SessionId = std::uint16_t;
using ShortName = std::string;
using DataTypeName = std::string;
using EventTypeName = std::string;
using StatisticInfo = std::map<StatisticKind, uint64_t>;

const size_t MAX_EVENT_SUB_COUNT {1000U};
InstanceId const UNDEFINED_INSTANCEID {"65534"};
constexpr ServiceId UNDEFINED_SERVICEID {0xFFFEU};
constexpr std::uint32_t UNDEFINED_UID {0xFFFFFFFFU};
constexpr EntityId UNDEFINED_ENTITYID {0xFFFFFFFEU};
InstanceId const ANY_INSTANCEID {"65535"};
constexpr EntityId ANY_METHODID {0xFFFFU};
constexpr ServiceId ANY_SERVICEID {0xFFFFU};
constexpr MajorVersionId ANY_MAJOR_VERSIONID {0xFFU};
constexpr MinorVersionId ANY_MINOR_VERSIONID {0xFFFFFFFFU};
constexpr size_t DEFAULT_EVENT_CACHESIZE {10U};
constexpr std::uint32_t DEFAULT_LOG_LIMIT {500U};
constexpr std::uint32_t LOG_LIMIT_SECOND_60 {60000U}; // 60s
const ara::godel::common::log::LogLimitConfig LOG_LIMIT_CONFIG {
    DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT};
constexpr uint32_t MBUF_POOL_MAX_BLK_NUM {32000U};
constexpr uint32_t MBUF_POOL_MAX_BLK_SIZE {16777216U};
std::string const UNDEFINED_SERVICE_NAME {"UNDEFINED_SERVICE_NAME"};
std::string const UNDEFINED_QOS_PROFILE {"UNDEFINED_QOS_PROFILE"};
NetworkIp const UNDEFINED_NETWORK{"UNDEFINED_NETWORK", false};
NetworkIp const BUILTIN_APP_CLIENT_NETWORK{"BUILTIN_APP_CLIENT_FLAG", false};
NetworkIp const BUILTIN_APP_SERVER_NETWORK{"BUILTIN_APP_SERVER_FLAG", false};
enum class ReturnCode: uint8_t {
    OK = 0U,
    ERROR = 1U,
    TIMEOUT = 2U
};

struct Result {
    std::shared_ptr<uint8_t> data;
    size_t length;
};

enum class EventSubscriptionState : uint8_t {
    kSubscribed,
    kNotSubscribed,
    kSubscriptionPending
};

enum class MethodState : uint8_t {
    kMethodOnline,
    kMethodOffline
};

enum class MethodCallProcessingMode: uint8_t {
    kPoll = 0U,
    kEvent = 1U,
    kEventSingleThread = 2U
};

enum class ListenerStatus: uint8_t {
    OK          = 0U,
    PARAM_ERROR = 1U,
    REPEATED    = 2U,
    INVALID     = 3U
};
template <typename T>
using SampleContainer = std::deque<T>;
using E2EResultCache = std::deque<vrtf::com::e2exf::Result>;
// 19-11 SWS_CM_00302
class InstanceIdentifier {
public:
    static InstanceId const Any; // For 1803 SWS_CM_00302
    ~InstanceIdentifier(void) = default;
    explicit InstanceIdentifier(const ara::core::StringView& value);
    ara::core::StringView ToString() const;
    bool operator==(const InstanceIdentifier& other) const;
    bool operator<(const InstanceIdentifier& other) const;
    InstanceIdentifier& operator=(const InstanceIdentifier& other);
    InstanceIdentifier(const InstanceIdentifier& other);
    const InstanceId GetInstanceIdString() const noexcept
    {
        return stringId_;
    }

private:
    InstanceId stringId_ = UNDEFINED_INSTANCEID;
};

class HandleType {
public:
    HandleType(const ServiceId& serviceId, const InstanceId& id, const DriverType& driver)
        : serviceId_(serviceId), id_(id), driver_(driver) {}
    // SWS_CM_00317
    HandleType(const HandleType &other) = default;
    HandleType& operator=(const HandleType& other) = default;

    // SWS_CM_00318
    HandleType(HandleType &&other) = default;
    HandleType& operator=(HandleType &&other) = default;

    ~HandleType() = default;
    DriverType GetDriver() const
    {
        return driver_;
    }
    InstanceIdentifier GetInstanceId() const
    {
        return InstanceIdentifier(ara::core::StringView(id_.c_str()));
    }
    ServiceId GetServiceId() const
    {
        return serviceId_;
    }

    bool operator<(const HandleType &other) const
    {
        return (driver_ < other.driver_) || (driver_ == other.driver_ && serviceId_ < other.serviceId_) ||
            (driver_ == other.driver_ && serviceId_ == other.serviceId_ && id_ < other.id_);
    }

    bool operator==(const HandleType &other) const
    {
        return (id_ == other.id_) && (driver_ == other.driver_) && (serviceId_ == other.serviceId_);
    }
private:
    ServiceId serviceId_ {UNDEFINED_SERVICEID};
    InstanceId id_;
    DriverType driver_;
};

template<typename T>
using ServiceHandleContainer = std::vector<T>;
using SubscriptionStateChangeHandler = std::function<void(EventSubscriptionState)>;
using MethodStateChangeHandler = std::function<void(MethodState)>;
class FindServiceHandle {
public:
    FindServiceHandle() = default;
    ~FindServiceHandle(void) = default;
    bool operator==(const FindServiceHandle& other) const;
    bool operator<(const FindServiceHandle& other) const;
    FindServiceHandle(const FindServiceHandle& other) = default;
    FindServiceHandle& operator=(const FindServiceHandle& other) = default;

    // Internal class!!! Prohibit to use by Application!!!!
    explicit FindServiceHandle(std::uint32_t uid) : uid_(uid){}
    // Internal class!!! Prohibit to use by Application!!!!
    std::uint32_t GetUID() const;

private:
    std::uint32_t uid_ = vrtf::vcc::api::types::UNDEFINED_UID;
};

template<typename T>
using FindServiceHandler = std::function<void(ServiceHandleContainer<T>, FindServiceHandle)>;
class ApplicationName {
public:
    static std::shared_ptr<ApplicationName>& GetInstance()
    {
        static std::shared_ptr<ApplicationName> instance{std::make_shared<ApplicationName>()};
        return instance;
    }

    void SetName(const std::string& inputApplicationName);
    std::string GetName()
    {
        return applicationName_;
    }
private:
    std::string applicationName_;
};
class VersionInfo {
public:
    VersionInfo(MajorVersionId major, MinorVersionId serviceMinor)
        : majorVersion_(major), serviceMinorVersion_(serviceMinor) {}
    VersionInfo(){}
    ~VersionInfo(void) = default;
    void SetMajorVersion(const MajorVersionId& major)
    {
        majorVersion_ = major;
    }
    void SetServiceMinorVersion(const MinorVersionId& serviceMinor)
    {
        serviceMinorVersion_ = serviceMinor;
    }
    void SetInstanceMinorVersion(const MinorVersionId& instanceMinor);
    void SetVersionDrivenFindBehavior(std::string const &behavior);
    VersionDrivenFindBehavior GetVersionDrivenFindBehavior() const;
    MajorVersionId GetMajorVersion() const
    {
        return majorVersion_;
    }
    MinorVersionId GetServiceMinorVersion() const
    {
        return serviceMinorVersion_;
    }
    MinorVersionId GetInstanceMinorVersion() const;
    bool operator==(const VersionInfo& other) const
    {
        return (majorVersion_ == other.GetMajorVersion()) && (serviceMinorVersion_ == other.GetServiceMinorVersion());
    }

    bool operator<(const VersionInfo& other) const
    {
        if (majorVersion_ < other.GetMajorVersion()) {
            return true;
        } else if (majorVersion_ == other.GetMajorVersion()) {
            return serviceMinorVersion_ < other.GetServiceMinorVersion();
        } else {
            // do nothing
        }

        return false;
    }
    VersionInfo(const VersionInfo& other) = default;
    VersionInfo& operator=(const VersionInfo& other) = default;
    void ResetInstanceConfig();
private:
    MajorVersionId majorVersion_ {ANY_MAJOR_VERSIONID};
    MinorVersionId serviceMinorVersion_ {ANY_MINOR_VERSIONID};
    MinorVersionId instanceMinorVersion_ {ANY_MINOR_VERSIONID};
    VersionDrivenFindBehavior findBehavior_ {VersionDrivenFindBehavior::EXACT_OR_ANY_MINOR_VERSION};
};
class SampleTimeInfo {
public:
    void SetReceiveModule(const std::uint8_t& module)
    {
        receiveModule_ = module;
    }

    const std::uint8_t& GetReceiveModule() const
    {
        return receiveModule_;
    }

    void SetPlogId(const std::uint64_t& id)
    {
        plogId_ = id;
    }

    const std::uint64_t& GetPlogId() const
    {
        return plogId_;
    }

    void SetTakeTime(const std::uint64_t& time)
    {
        takeTime_ = time;
    }

    const std::uint64_t& GetTakeTime() const
    {
        return takeTime_;
    }

    void SetServerSendTime(const timespec& time)
    {
        serverSendTime_ = time;
    }

    const timespec& GetServerSendTime() const
    {
        return serverSendTime_;
    }
private:
    std::uint8_t receiveModule_ = 0U;
    std::uint64_t plogId_ = 0U;
    std::uint64_t takeTime_ {0U};
    timespec serverSendTime_ {0, 0};
};
class SampleInfo {
public:
    void SetSampleId(const std::uint64_t& id)
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&id);
#endif
        sampleId_ = id;
    }
    const std::uint64_t& GetSampleId() const
    {
        return sampleId_;
    }
    void SetE2EResult(const vrtf::com::e2exf::Result& e2eResult)
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&e2eResult);
#endif
        e2eResult_ = e2eResult;
    }
    const vrtf::com::e2exf::Result GetE2EResult() const
    {
        return e2eResult_;
    }
private:
    std::uint64_t sampleId_;
    vrtf::com::e2exf::Result e2eResult_ = vrtf::com::e2exf::Result(vrtf::com::e2exf::ProfileCheckStatus::kNotAvailable,
                                                                   vrtf::com::e2exf::SMState::kStateMDisabled);
};

namespace internal {
class SampleInfoImpl {
public:
    SampleInfoImpl() = default;
    ~SampleInfoImpl() = default;
    std::shared_ptr<vrtf::vcc::utils::PlogInfo> plogInfo_ = nullptr;
    bool isEvent_ = true;
    timespec sendTime_ {0, 0};
};
}
namespace reserve {
    std::string const AP_NODE_NAME_SPACE{"/Huawei/AP"};
    std::string const PHM_NODE_NAME{"PlatformHealthManagement"};
    std::string const EM_NODE_NAME{"ExecutionManagement"};
    // The current external service domain use the domain ID 0x02 by PHM/EM/Maintaind
    constexpr int16_t EXTERNAL_SERVICE_DOMAIN_ID = 0x02;
    // The current maintaind service domain use the domain ID 0x03 by Rtftools
    constexpr int16_t MAINTAIND_SERVICE_DOMAIN_ID = 0x03;
    // The current internal service domain use the domain ID 0x04 by PHM/EM
    constexpr int16_t INTERNAL_SERVICE_DOMAIN_ID = 0x04;
}

class ServiceDiscoveryInfo {
public:
    ServiceDiscoveryInfo()
        : serviceId_(0U), instanceId_("0"), methodMode_(MethodCallProcessingMode::kEvent) {}
    virtual ~ServiceDiscoveryInfo() = default;

    ServiceId GetServiceId() const noexcept
    {
        return serviceId_;
    }
    void SetServiceId(ServiceId id)
    {
        serviceId_ = id;
    }
    InstanceId GetInstanceId() const noexcept
    {
        return instanceId_;
    }
    uint16_t GetU16InstanceId() const noexcept
    {
        return instanceIdU16_;
    }
    bool SetInstanceId(const InstanceId& id);
    void SetMethodCallProcessingMode(const MethodCallProcessingMode& mode)
    {
        methodMode_ = mode;
    }

    void SetVersion(const VersionInfo& version)
    {
        version_ = version;
    }

    VersionInfo GetVersion() const
    {
        return version_;
    }

    const NetworkIp& GetNetwork() const
    {
        return network_;
    }

    void SetNetwork(const NetworkIp& network)
    {
        network_ = network;
    }

    void SetConfigInfoByApi(const bool setConfigInfoByApi)
    {
        setConfigInfoByApi_ = setConfigInfoByApi;
    }
    bool IsSetConfigInfoByApi() const
    {
        return setConfigInfoByApi_;
    }
    virtual DriverType GetDriverType() const = 0;
    std::string GetDiscoveryUUIDInfo() const
    {
        const std::map<DriverType, std::string> driverTypeMap {
        std::map<DriverType, std::string>::value_type(DriverType::PROLOCTYPE, "PROLOC"),
        std::map<DriverType, std::string>::value_type(DriverType::DDSTYPE, "DDS"),
        std::map<DriverType, std::string>::value_type(DriverType::SOMEIPTYPE, "SOMEIP")
        };
        std::stringstream discoveryUUID;
        discoveryUUID << serviceId_ << "." << instanceId_ << "." << driverTypeMap.at(GetDriverType());
        return discoveryUUID.str();
    }
private:
    ServiceId serviceId_;
    InstanceId instanceId_;
    uint16_t instanceIdU16_ = 65534U;
    MethodCallProcessingMode methodMode_;
    VersionInfo version_;
    NetworkIp network_ = UNDEFINED_NETWORK;
    bool setConfigInfoByApi_ = false;
};
}
}
}
}
#endif

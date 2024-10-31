/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Stats header
 * Create: 2021-06-10
 */
#ifndef VRTF_VCC_UTILS_STATS_H
#define VRTF_VCC_UTILS_STATS_H

#include <string>
#include <atomic>

#include "vrtf/vcc/api/types.h"

namespace vrtf {
namespace vcc {
namespace utils {
namespace stats {
class EntityIdentifier {
public:
    using ServiceId = vcc::api::types::ServiceId;
    using InstanceId = vcc::api::types::InstanceId;
    using EntityId = vcc::api::types::EntityId;

    explicit EntityIdentifier(ServiceId const serviceId, InstanceId const instanceId, EntityId const entityId)
        : serviceId_(serviceId), instanceId_(instanceId), entityId_(entityId) {}

    ~EntityIdentifier() = default;

    bool operator<(EntityIdentifier const &other) const;

    bool operator==(EntityIdentifier const &other) const;

    std::string ToString() const noexcept;

private:
    ServiceId serviceId_;
    InstanceId instanceId_;
    EntityId entityId_;
};

class Stats {
public:
    explicit Stats(EntityIdentifier const &id) : entityIdentifier_(id), appName_("")
    {
        appName_ = vrtf::vcc::api::types::ApplicationName::GetInstance()->GetName();
    }

    virtual ~Stats() = default;

    virtual EntityIdentifier GetEntityIdentifier() const noexcept;

    void SetEntityName(vrtf::vcc::api::types::DriverType const type, std::string const &name) noexcept;

    std::map<vrtf::vcc::api::types::DriverType, std::string> GetEntityName() const noexcept;

    virtual std::string ToPlogString() noexcept = 0;
    virtual std::string ToPrintString() const noexcept = 0;
protected:
    using CountType = std::atomic_uint64_t;
    EntityIdentifier entityIdentifier_;
    std::string appName_;
    std::map<vrtf::vcc::api::types::DriverType, std::string> mapEntityName_ {};
};

class EventProxyStats : public Stats {
public:
    explicit EventProxyStats(EntityIdentifier const &id);

    ~EventProxyStats() override = default;

    std::string ToPlogString() noexcept override;
    std::string ToPrintString() const noexcept override;
    void SetUsingCallbackFlag(bool const flag) noexcept;

    void IncFromProtocolCallbackCount() noexcept;

    void IncToUserCallbackCount() noexcept;

    void IncFromProtocolCount(CountType const &count) noexcept;

    uint64_t GetFromProtocolCount() const noexcept;

    void IncToUserCount() noexcept;

    uint64_t GetToUserCount() const noexcept;

    void IncExceededBufferCount(CountType const &count) noexcept;

    void IncDeserializationFailedCount() noexcept;

    void IncFromProtByteCount(CountType const &byte) noexcept;

    uint64_t GetFromProtByteCount() const noexcept;

    void IncExceededBufferByteCount(CountType const &byte) noexcept;

    uint64_t GetExceededBufferByteCount() const noexcept;

    void IncDeserializationFailedByteCount(CountType const &byte) noexcept;

    uint64_t GetDeserializationFailedByteCount() const noexcept;

    void IncGetDataFinishCount() noexcept;

private:
    std::atomic_bool usingCallback_ {false};

    CountType cbFromProt_ {0U};
    CountType cbToUser_ {0U};
    CountType fromProtCount_ {0U};
    CountType fromProtByteCount_ {0U};
    CountType toUsrCount_ {0U};
    CountType receiveDataFinish_ {0U};
    CountType exceededBufferCount_ {0U};
    CountType exceededBufferByteCount_ {0U};
    CountType deserializationFailedCount_ {0U};
    CountType deserializationFailedByteCount_ {0U};
};

class EventSkeletonStats : public Stats {
public:
    explicit EventSkeletonStats(EntityIdentifier const &id);

    ~EventSkeletonStats() override = default;

    std::string ToPlogString() noexcept override;
    std::string ToPrintString() const noexcept override;
    void SetDpRawDataFlag() noexcept;

    void IncFromUserCount() noexcept;

    void IncToDDSCount() noexcept;

    void IncToSomeipCount() noexcept;

    void IncToDDSByteCount(CountType const &byte) noexcept;

    void IncToSomeipByteCount(CountType const &byte) noexcept;

    void IncTrafficControlledCount() noexcept;

    void IncDDSAllocateFailedCount() noexcept;

    void IncSomeipAllocateFailedCount() noexcept;

    void IncRawDataGetAvailabeLenthFailedCount() noexcept;

    void IncRawDataGetSerializeBufferFailedCount() noexcept;

private:
    bool isDpRawDate_ {false};

    CountType fromUsrCount_ {0U};
    CountType toDDSCount_ {0U};
    CountType toDDSByteCount_ {0U};
    CountType toSomeipCount_ {0U};
    CountType toSomeipByteCount_ {0U};
    CountType trafficControlledCount_ {0U};
    CountType ddsAllocateFailedCount_ {0U};
    CountType someipAllocateFailedCount_ {0U};
    CountType rawDataGetAvailabeLenthFailedCount_ {0U};
    CountType rawDataGetSerializeBufferFailedCount_ {0U};
};

class MethodProxyStats : public Stats {
public:
    explicit MethodProxyStats(EntityIdentifier const &id) : Stats(id) {}

    ~MethodProxyStats() override = default;

    std::string ToPlogString() noexcept override;
    std::string ToPrintString() const noexcept override;
    void IncFromUsrCount() noexcept {static_cast<void>(++fromUsrCount_);}
    void IncServiceNotAvailable() noexcept {static_cast<void>(++serviceNotAvailable_);}
    void IncNoInitialized() noexcept { static_cast<void>(++noInitialize_);}
    void IncSerializeFailed() noexcept { static_cast<void>(++serializeFail_);}
    void IncDdsAllocateFailed() noexcept { static_cast<void>(++ddsAllocateFail_);}
    void IncDdsInvalidData() noexcept {static_cast<void>(++ddsInvalidData_);}
    void IncDdsCheckServerFail() noexcept {static_cast<void>(++ddsCheckServerFail_);}
    void IncE2eProtectFail() noexcept {static_cast<void>(++e2eProtectFail_);}
    void IncSecocAddFail() noexcept { static_cast<void>(++secocAddFail_);}
    void IncDdsWriteFail() noexcept {static_cast<void>(++ddsWriteFail_);}
    void IncToProtocolCount() noexcept {static_cast<void>(++toProtocolCount_);}
    void IncSomeipCreateRequestMsgFail() noexcept { static_cast<void>(++someipCreateRequestMsgFail_);}
    void IncSomeipRequestFail() noexcept { static_cast<void>(++someipRequestFail_);}
    void IncFromProtocolCount() noexcept { static_cast<void>(++fromProtocolCount_);}
    void IncToUsrCount() noexcept { static_cast<void>(++toUsrCount_);}
    void IncE2eCheckFail() noexcept { static_cast<void>(++e2eCheckFail_);}
    void IncTlsShakeFail() noexcept { static_cast<void>(++tlsShakeFail_);}
    void IncSomeipCheckMsgFail() noexcept { static_cast<void>(++someipCheckMsgFail_);}
    void IncSomeipWrongSessionId() noexcept { static_cast<void>(++someipWrongSessionId_);}
    bool HasDataReceive() const noexcept {return (fromUsrCount_ != 0);}
private:
    void AddExceptionStatistic(std::stringstream& ss, bool isDds, bool isSomeip) const;
    // request
    CountType fromUsrCount_ {0U};
    CountType toProtocolCount_ {0U};

    CountType serviceNotAvailable_ {0U};
    CountType noInitialize_ {0U};
    CountType serializeFail_ {0U};
    CountType e2eProtectFail_ {0U};
    CountType secocAddFail_ {0U};

    CountType ddsAllocateFail_ {0U};
    CountType ddsInvalidData_ {0U};
    CountType ddsCheckServerFail_ {0U};
    CountType ddsWriteFail_ {0U};

    CountType someipCreateRequestMsgFail_ {0U};
    CountType someipRequestFail_ {0U};

    // reply
    CountType fromProtocolCount_ {0U};
    CountType toUsrCount_ {0U};
    CountType e2eCheckFail_ {0U};
    CountType tlsShakeFail_ {0U};
    CountType someipCheckMsgFail_ {0U};
    CountType someipWrongSessionId_ {0U};

};

class MethodSkeletonStats : public Stats {
public:
    explicit MethodSkeletonStats(EntityIdentifier const &id) : Stats(id) {}

    ~MethodSkeletonStats() override = default;

    std::string ToPlogString() noexcept override;
    std::string ToPrintString() const noexcept override;
    void IncFromDdsCount() noexcept { static_cast<void>(++fromDdsCount_);}
    void IncFromSomeipCount() noexcept { static_cast<void>(++fromSomeipCount_);}
    void IncCallUserCallback() noexcept { static_cast<void>(++callUserCallback_);}
    void IncCallE2EErrorHandle() noexcept { static_cast<void>(++callE2EErrorHandle_);}
    void IncFromUsrCount() noexcept { static_cast<void>(++fromUsrCount_);}
    void IncInvalidData() noexcept { static_cast<void>(++invalidData_);}
    void IncWrongInstance() noexcept { static_cast<void>(++wrongInstance_);}
    void IncTrafficControlledCount() noexcept { static_cast<void>(++trafficControlledCount_);}
    void IncToDdsCount() noexcept { static_cast<void>(++toDdsCount_);}
    void IncToSomeipCount() noexcept { static_cast<void>(++toSomeipCount_);}
    void IncSomeipCheckMsgFail() noexcept { static_cast<void>(++someipCheckMsgFail_);}
    bool HasDataReceive() const noexcept {return (fromDdsCount_ != 0 || fromSomeipCount_ != 0);}
    void IncDdsAllocateFailed() noexcept { static_cast<void>(++ddsAllocateFail_);}
private:
    CountType fromDdsCount_ {0U};
    CountType fromSomeipCount_ {0U};
    CountType callUserCallback_ {0U};
    CountType callE2EErrorHandle_ {0U};

    CountType fromUsrCount_ {0U};
    CountType toDdsCount_ {0U};
    CountType toSomeipCount_ {0U};

    CountType invalidData_ {0U};
    CountType wrongInstance_ {0U};
    CountType trafficControlledCount_ {0U};
    CountType someipCheckMsgFail_ {0U};
    CountType ddsAllocateFail_{0U};
};
} // namespace vrtf
} // namespace vcc
} // namespace utils
} // namespace stats
#endif // VRTF_VCC_UTILS_STATS_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Define types in communication mannger
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_API_INTERNAL_DRIVERTYPE_H
#define VRTF_VCC_API_INTERNAL_DRIVERTYPE_H
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "vrtf/vcc/api/param_struct_typse.h"
#include "vrtf/vcc/api/method_error.h"
#include "vrtf/vcc/api/raw_buffer_helper.h"
#include "ara/hwcommon/log/log.h"
#include "vrtf/vcc/serialize/serialize_config.h"
struct SecOCHandler;
namespace vrtf {
namespace vcc {
namespace api {
namespace types {
struct ISignal {
    using LengthType = std::size_t;

    std::string name;
    std::string type;
    bool dynamic;
    LengthType startPos;
    LengthType length; // bits
    std::string rawByteOrder;
    vrtf::serialize::ByteOrder byteOrder;
};

class Pdu {
public:
    using LengthType = std::size_t;

    explicit Pdu(std::string shortName = "", Pdu::LengthType const len = 0U, const bool dynamic = false,
        std::vector<std::shared_ptr<ISignal>> iSignals = {}, const bool isCheck = true);
    ~Pdu() = default;
    std::shared_ptr<const ISignal> operator[](std::size_t const &idx) const;
    bool IsValid() const;
    std::string Name() const;
    bool Dynamic() const;
    LengthType Length() const;
    LengthType LengthWithoutDynamic() const;
    std::size_t ISignalCount() const;
    void SetRealDynamicLength(std::size_t payloadSize);
    LengthType GetRealDynamicLength() const;
    LengthType PayloadSize() const;
    void PayloadSize(std::size_t payloadSize);
    // roslike config pdu is not check
    bool IsCheck() const;

private:
    using IntervalType = std::pair<LengthType, LengthType>; // Interval type : [a, b)
    std::string name_;
    LengthType length_; // bytes
    LengthType lengthOfBits_; // bits
    LengthType nonDynamicLength_ {lengthOfBits_}; // bits
    LengthType realDynamicLength_ {0U}; // bits
    bool dynamic_;
    std::vector<std::shared_ptr<ISignal>> iSignals_;

    std::vector<IntervalType> intervals_;
    bool hasInvalidSignal_ {false};
    std::uint16_t dynamicSignalCount_ {0U};
    bool isValid_ {true};
    std::shared_ptr<ISignal> lastISignal_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    size_t payloadSize_;
    bool isCheck_;
    void PrepareCheckResource();
    bool IsOverlaped() const;
    bool IsDynamicMatched();
    bool CheckPdu();
    std::vector<IntervalType> GenIntervals(std::shared_ptr<ISignal> const &iSignal);
    static size_t Lsb2Msb(size_t startBit, size_t signalLen);
};

struct FileOwner {
    std::string user;
    std::string group;
};
class ResourceAttr {
public:
    ResourceAttr() = default;
    explicit ResourceAttr(const FileOwner& owner);
    virtual ~ResourceAttr() = default;
    void SetFileOwner(const FileOwner& owner);
    FileOwner GetFileOwner() const;
private:
    FileOwner fileOwner_;
};

enum class ResourceType : uint32_t {
    SHM
};

class ResourcePara {
public:
    explicit ResourcePara(const ResourceType &type);
    virtual ~ResourcePara() = default;
    ResourceType GetType() const noexcept;
private:
    ResourceType type_;
};

class ShmObject : public ResourcePara {
public:
    ShmObject(const ResourceType &type, const std::string &para);
    ~ShmObject() override = default;
    std::string GetValue() const noexcept;
private:
    std::string para_;
};

using ResourceCreateHandler = std::function<bool(const std::shared_ptr<ResourcePara>&)>;

struct SecOCHandlerPair {
    SecOCHandler* authHandler;
    SecOCHandler* deauthHandler;
};

enum class SecOCSecureComType : uint8_t {
    SECOC_CALL = 0U,
    SECOC_RETURN = 1U,
    SECOC_EVENT = 2U
};

// tools
struct EventE2EConfigInfo {
    uint16_t serviceId;
    std::string instanceId;
    std::string eventTopicName;
    uint16_t eventId {0U};
    uint32_t entityId;
    vrtf::vcc::api::types::DriverType drivertype;
    std::shared_ptr<const vrtf::com::e2exf::E2EXf_CMConfig> e2eXfCmConfig;
};

class EntityInfo {
public:
    EntityInfo() = default;
    virtual ~EntityInfo() = default;
    void SetEntityId(const EntityId& id) { id_ = id; }
    EntityId GetEntityId() const { return id_; }
    ServiceId GetServiceId() const { return serviceId_; }
    void SetServiceId(ServiceId id) { serviceId_ = id; }
    InstanceId GetInstanceId() const { return instanceId_; }
    uint16_t GetU16InstanceId() const { return u16InstanceId_; }
    bool SetInstanceId(const InstanceId& id);
    void SetVersion(const VersionInfo& version) { version_ = version; }
    VersionInfo GetVersion() const { return version_; }
    void SetShortName(const ShortName& shortName) { shortName_ = shortName; }
    ShortName GetShortName() const { return shortName_; }
    void SetServiceShortName(const ShortName& serviceShortName) { serviceShortName_ = serviceShortName; }
    ShortName GetServiceShortName() const { return serviceShortName_; }
    void SetDataStatStatus(bool status) { isEnableDataStat_ = status; }
    bool GetDataStatStatus() const { return isEnableDataStat_; }
    void SetInstanceShortName(const std::map<bool, ShortName>& instanceShortName)
    {
        instanceShortName_ = instanceShortName;
    }
    std::map<bool, ShortName> GetInstanceShortName() const { return instanceShortName_; }
    const NetworkIp& GetNetwork() const;
    void SetNetwork(const NetworkIp& network);
    void SetConfigInfoByApi(const bool setConfigInfoByApi) { setConfigInfoByApi_ = setConfigInfoByApi; }
    bool IsSetConfigInfoByApi() const { return setConfigInfoByApi_; }
    std::string GetEntityIdentifier() const {
        return std::to_string(serviceId_) + "_" + instanceId_ + "_" + std::to_string(id_); }

    /**
     * @brief  Set E2EInfo which will used in driver
     *
     * @param[in] e2eObject   The pointer points to E2EXf_Object object
     * @param[in] e2eXfCmConfig  The pointer points to E2EXf_CMConfig object
     */
    void SetE2EHandler(const std::shared_ptr<vrtf::com::e2exf::E2EXf_Handler>& e2eHandler) { e2eHandler_ = e2eHandler; }

    /**
     * @brief  Get E2EXf_Handler which contains used configuration and operations
     *
     * @return std::shared_ptr<vrtf::com::e2exf::E2EXf_Handler>   The pointer points to E2EXf_Handler object
     */
    std::shared_ptr<vrtf::com::e2exf::E2EXf_Handler> GetE2EHandler() const { return e2eHandler_; }

    /**
     * @brief Set a ResourceAttr
     *
     * @param[in] resourceAttr File's user and group
     */
    void SetResourceAttr(const ResourceAttr& resourceAttr) noexcept;

    /**
     * @brief Get a ResourceAttr
     *
     * @return ResourceAttr File's user and group
     */
    ResourceAttr GetResourceAttr() const noexcept;

    void SetResourceCreateHandler(const ResourceCreateHandler &handler);

    ResourceCreateHandler GetResourceCreateHandler() const noexcept;
    /**
     * @brief Get a E2EXfCMConfig
     *
     * @return Collection of all E2E configurations
     */
    std::shared_ptr<const vrtf::com::e2exf::E2EXf_CMConfig> GetE2EXfCMConfig() const
    {
        if (e2eHandler_ != nullptr) {
            return e2eHandler_->GetCMConfig();
        }
        return nullptr;
    }

    /**
     * @brief If allow to set e2e errorcode in method proxy
     *
     * @return bool
     *      @retval false   not allowed
     *      @retval true    allowed
     */
    bool IsSettingResponseE2EErrc() const { return isSetResponseE2EErrc_; }

    /**
     * @brief  Set if allow to set e2e errorcode in method proxy
     *
     * @param[in] isSetResponseE2EErrc   if allow to set e2e errorcode in method proxy
     */
    void SetIsSettingResponseE2EErrc(bool isSetResponseE2EErrc) { isSetResponseE2EErrc_ = isSetResponseE2EErrc; }
    virtual vrtf::vcc::api::types::DriverType GetDriverType() const noexcept = 0;
    std::string GetDriverTypeStr() const
    {
        const std::map<DriverType, std::string> driverTypeMap {
            std::map<DriverType, std::string>::value_type(DriverType::PROLOCTYPE, "PROLOC"),
            std::map<DriverType, std::string>::value_type(DriverType::DDSTYPE, "DDS"),
            std::map<DriverType, std::string>::value_type(DriverType::SOMEIPTYPE, "SOMEIP")
        };
        return driverTypeMap.at(GetDriverType());
    }
private:
    EntityId id_ = ANY_METHODID;
    ServiceId serviceId_ {0U};
    InstanceId instanceId_ = "65534";
    uint16_t u16InstanceId_ = 65534U;
    VersionInfo version_;
    ShortName shortName_;
    std::map<bool, ShortName> instanceShortName_;
    NetworkIp network_ = UNDEFINED_NETWORK;
    bool setConfigInfoByApi_ {false};
    bool isEnableDataStat_ {true};
    vrtf::serialize::SerializationType serializationType_ = vrtf::serialize::SerializationType::CM;
    std::shared_ptr<vrtf::com::e2exf::E2EXf_Handler> e2eHandler_ {nullptr};
    ResourceAttr resourceAttr_;
    // If setting E2E Error Code of received response checking result in Proxy
    bool isSetResponseE2EErrc_ {false};
    ResourceCreateHandler resHandler_;
    ShortName serviceShortName_;
};

class EventInfo : public EntityInfo {
public:

    EventInfo() = default;
    ~EventInfo() override = default;

    void SetIsField(bool isField) { isField_ = isField; }
    bool GetIsField() const { return isField_; }
    void SetCacheSize(const size_t size) { cacheSize_ = size; }
    size_t GetCacheSize() const { return cacheSize_; }
    void SetDataTypeName(const DataTypeName& dataTypeName) { dataTypeName_ = dataTypeName; }
    DataTypeName GetDataTypeName() const { return dataTypeName_; }
    inline void SetDpRawDataFlag(bool isDpRawData) { isDpRawData_ = isDpRawData; }
    inline bool GetDpRawDataFlag() const { return isDpRawData_; }
    inline void SetRawMemoryFlag(bool isRawMemory) { isRawMemory_ = isRawMemory; }
    inline bool GetRawMemoryFlag() const noexcept { return isRawMemory_; }
    inline void SetThreadMode(const vcc::api::types::ThreadMode& mode) noexcept { threadMode_ = mode; }
    inline vcc::api::types::ThreadMode GetThreadMode() const noexcept { return threadMode_; }
    void SetSerializeConfig(vrtf::serialize::SerializeConfig const &config) { serializeConfig_ = config; }
    vrtf::serialize::SerializeConfig GetSerializeConfig() const { return serializeConfig_; }
    inline void SetRawBufferHelper(const std::shared_ptr<vrtf::vcc::RawBufferHelper>& rawBufferHelper)
    {
        rawBufferHelper_ = rawBufferHelper;
    }
    inline std::shared_ptr<vrtf::vcc::RawBufferHelper> GetRawBufferHelper() const { return rawBufferHelper_; }

    /**
     * @brief Set true if the public API binding ReceivedHandler and GetNewSample, using in the RTFCOM.
     *        That is the public API of RTFCM is the combinding of ReceivedHandler and GetNewSample.
     *
     * @param[in] isCombined  If it is the combinding API
     */
    void CombindReceivedHdlAndGetSample(const bool isCombined) { isCombindReceivdHdlAndGetSample_ = isCombined; }

    /**
     * @brief Get whether it is used in the combination API, RTFCOM.
     *
     * @retval true  The EventInfo is used in combination mode
     * @retval false The EventInfo is not used in combination  mode
     */
    bool IsCombindReceivdHdlAndGetSample() const { return isCombindReceivdHdlAndGetSample_; }
    void SetPdu(std::shared_ptr<vrtf::vcc::api::types::Pdu> const &pdu)
    {
        if (pdu != nullptr) {
            pdu_ = std::make_shared<vrtf::vcc::api::types::Pdu>(*pdu);
        }
        isSignalBased_ = true;
    }
    std::shared_ptr<vrtf::vcc::api::types::Pdu> GetPdu() const
    {
        return pdu_;
    }
    void IsSignalBased(bool const isSignalBased)
    {
        isSignalBased_ = isSignalBased;
    }
    bool IsSignalBased() const
    {
        return isSignalBased_;
    }
    void SetSecOCHandler(const SecOCHandlerPair &secocHandler)
    {
        secocHandlers_ = secocHandler;
    }
    SecOCHandlerPair GetSecOCHandler() const
    {
        return secocHandlers_;
    }
    vrtf::serialize::SerializationNode GetSerializationNode() const
    {
        return topSerializationNode_;
    }
    void SetSerializationNode(vrtf::serialize::SerializationNode const &topSerializationNode)
    {
        topSerializationNode_ = topSerializationNode;
    }
    std::string GetEventUUIDInfo() const
    {
        std::stringstream eventUUID;
        eventUUID << GetServiceId() << "." << GetInstanceId() << "." << GetShortName() << "." << GetEntityId() << "."
            << GetDriverTypeStr();
        return eventUUID.str();
    }
private:
    bool isField_ {false};
    size_t cacheSize_ {DEFAULT_EVENT_CACHESIZE};
    DataTypeName dataTypeName_;
    bool isDpRawData_ {false};
    bool isRawMemory_ {false};
    vrtf::serialize::SerializeConfig serializeConfig_;
    std::shared_ptr<vrtf::vcc::RawBufferHelper> rawBufferHelper_;
    bool isCombindReceivdHdlAndGetSample_ {false};
    std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_ {nullptr};
    bool isSignalBased_ {false};

    SecOCHandlerPair secocHandlers_ {nullptr, nullptr};
    vrtf::serialize::SerializationNode topSerializationNode_;
    vcc::api::types::ThreadMode threadMode_ = vcc::api::types::ThreadMode::EVENT;
};

class MethodInfo : public EntityInfo {
public:
    MethodInfo() = default;
    ~MethodInfo() = default;
    void SetFireAndForget(bool isFireAndForget) { isFireAndForget_ = isFireAndForget; }
    void SetMethodCallProcessingMode(const MethodCallProcessingMode methodMode) { methodMode_ = methodMode; }
    bool GetFireAndForget() const { return isFireAndForget_; }
    void SetIsField(bool isField) { isField_ = isField; }
    bool GetIsField() const { return isField_;}
    internal::MethodType GetMethodType() const { return methodType_; }
    void SetMethodType(const internal::MethodType methodType) { methodType_ = methodType; }
    void SetInputParameterList(const std::vector<std::string>& inputParameterList)
    {
        inputParameterList_ = inputParameterList;
    }
    std::vector<std::string> GetInputParameterList() const { return inputParameterList_; }
    void SetOutputParameterList(const std::vector<std::string>& outputParameterList)
    {
        outputParameterList_ = outputParameterList;
    }
    std::vector<std::string> GetOutputParameterList() const { return outputParameterList_; }
    void SetMethodReplyName(std::string const &replyName) { replyName_ = replyName; }
    std::string GetMethodReplyName() const { return replyName_; }
    void SetRequestSerializeConfig(const vrtf::serialize::SerializeConfig& config)
    {
        serializeRequestConfig_ = config;
    }
    vrtf::serialize::SerializeConfig GetRequestSerializeConfig() const { return serializeRequestConfig_; }
    void SetReplySerializeConfig(const vrtf::serialize::SerializeConfig& config) { serializeReplyConfig_ = config; }
    vrtf::serialize::SerializeConfig GetReplySerializeConfig() const { return serializeReplyConfig_; }
    std::deque<vrtf::vcc::api::types::internal::ErrorDomainInfo> GetMethodErrors() const { return errors_; }
    void SetMethodErrors(const std::deque<vrtf::vcc::api::types::internal::ErrorDomainInfo>& error)
    {
        errors_ = error;
    }

    std::string GetMethodUUIDInfo() const
    {
        // Composition format ServiceId.InstanceId.ShortName.MethodEntityId.DriverType
        std::stringstream methodUUID;
        methodUUID << GetServiceId() << "." << GetInstanceId() << "." << GetShortName() << "." << GetEntityId() << "."
            << GetDriverTypeStr();
        return methodUUID.str();
    }

    using SecOCHandlerContainerByComType = std::map<SecOCSecureComType, SecOCHandlerPair>;
    void SetSecOCHandler(const SecOCHandlerContainerByComType &secocHandlers)
    {
        secocHandlers_ = secocHandlers;
    }
    SecOCHandlerContainerByComType GetSecOCHandler() const
    {
        return secocHandlers_;
    }
    vrtf::serialize::SerializationNode GetRequestSerializationNode() const
    {
        return requestTopSerializationNode_;
    }
    void SetRequestSerializationNode(vrtf::serialize::SerializationNode const &topSerializationNode)
    {
        requestTopSerializationNode_ = topSerializationNode;
    }
    vrtf::serialize::SerializationNode GetReplySerializationNode() const
    {
        return replyTopSerializationNode_;
    }
    void SetReplySerializationNode(vrtf::serialize::SerializationNode const &topSerializationNode)
    {
        replyTopSerializationNode_ = topSerializationNode;
    }
private:
    bool isFireAndForget_ {false};
    MethodCallProcessingMode methodMode_ {vrtf::vcc::api::types::MethodCallProcessingMode::kEvent};
    bool isField_ {false};
    internal::MethodType methodType_ {internal::MethodType::GENERAL_METHOD};
    std::vector<std::string> inputParameterList_;
    std::vector<std::string> outputParameterList_;
    vrtf::serialize::SerializeConfig serializeRequestConfig_;
    vrtf::serialize::SerializeConfig serializeReplyConfig_;
    std::string replyName_;
    std::deque<vrtf::vcc::api::types::internal::ErrorDomainInfo> errors_;
    SecOCHandlerContainerByComType secocHandlers_;
    vrtf::serialize::SerializationNode requestTopSerializationNode_;
    vrtf::serialize::SerializationNode replyTopSerializationNode_;
};

class FieldInfo {
public:
    FieldInfo(){}
    ~FieldInfo(void) = default;
    void SetEventInfo(const std::shared_ptr<EventInfo>& event) { eventInfo_ = event; }
    void SetGetterMethodInfo(const std::shared_ptr<MethodInfo>& getter) { getterMethodinfo_ = getter; }
    void SetSetterMethodInfo(const std::shared_ptr<MethodInfo>& setter) { setterMethodinfo_ = setter; }
    EntityId GetEventEntityId() const
    {
        if (eventInfo_ != nullptr) {
            return eventInfo_->GetEntityId();
        }
        return UNDEFINED_ENTITYID;
    }
    EntityId GetGetterMethodEntityId() const
    {
        if (getterMethodinfo_ != nullptr) {
            return getterMethodinfo_->GetEntityId();
        } else {
            return vrtf::vcc::api::types::UNDEFINED_ENTITYID;
        }
    }
    EntityId GetSetterMethodEntityId() const
    {
        if (setterMethodinfo_ != nullptr) {
            return setterMethodinfo_->GetEntityId();
        } else {
            return vrtf::vcc::api::types::UNDEFINED_ENTITYID;
        }
    }
    std::shared_ptr<EventInfo> GetEventInfo() const { return eventInfo_; }
    std::shared_ptr<MethodInfo> GetGetterMethodInfo() const { return getterMethodinfo_; }
    std::shared_ptr<MethodInfo> GetSetterMethodInfo() const { return setterMethodinfo_; }
    void HasSetter(bool hasSetter) { hasSetter_ = hasSetter; }
    void HasGetter(bool hasGetter) { hasGetter_ = hasGetter; }
    void HasNotifier(bool hasNotifier) { hasNotifier_ = hasNotifier; }
    bool IsHasSetter() const { return hasSetter_; }
    bool IsHasGetter() const { return hasGetter_; }
    bool IsHasNotifier() const { return hasNotifier_; }

private:
    bool hasSetter_ {false};
    bool hasGetter_ {false};
    bool hasNotifier_ {false};

    std::shared_ptr<EventInfo> eventInfo_;
    std::shared_ptr<MethodInfo> getterMethodinfo_;
    std::shared_ptr<MethodInfo> setterMethodinfo_;
};
class MethodMsg {
public:
    MethodMsg()
        : entityId_(ANY_METHODID),
          sessionId_(UNDEFINED_SERVICEID),
          payload_(nullptr),
          size_(0U),
          type_(DriverType::INVALIDTYPE),
          instanceId_(UNDEFINED_INSTANCEID),
          e2eReceivedRequestCounter_(0U){}
    virtual ~MethodMsg() { payload_ = nullptr; }
    EntityId GetEntityId() const { return entityId_; }
    DriverType GetDriverType() const { return type_; }
    const uint8_t* GetPayload() const { return payload_; }
    std::size_t GetSize() const { return size_; }
    std::uint16_t GetSessionId() const { return sessionId_; }
    void SetEntityId(EntityId const entityId) { entityId_ = entityId; }
    void SetDriverType(DriverType const driverType) { type_ = driverType; }
    void SetPayload(uint8_t* payload) { payload_ = payload; }
    void SetSize(const size_t size) { size_ = size; }
    void SetSessionId(SessionId const sessionId) { sessionId_ = sessionId; }
    void SetMsgType(bool isErrorMsg) { isErrorMsg_ = isErrorMsg; }
    bool GetMsgType() const { return isErrorMsg_; }
    void SetTlsStatus(bool status) { isTlsShakeHandsOk_ = status; }
    bool GetTlsStatus() const { return isTlsShakeHandsOk_; }
    void SetTrafficCtrlFlag(const bool trafficControl) { trafficControl_ = trafficControl; }
    bool GetTrafficCtrlFlag() const { return trafficControl_; }
    void SetSerializeType(vrtf::serialize::SerializeType type) { serializeType_ = type; }
    vrtf::serialize::SerializeType GetSerializeType() const { return serializeType_; }
    /**
     * @brief Set source Id when using e2e profile 4m in server
     *
     * @param[in] sourceId the sourceId will be used in response
     */
    void SetE2ESourceId(const std::uint32_t sourceId) { e2eSourceId_ = sourceId; }

    /**
    * @brief Get the sourceId stored in msg when using profile 4m
    *
    * @return std::uint32_t         the sourceId stored in msg
    */
    std::uint32_t GetE2ESourceId() const { return e2eSourceId_; }

    /**
     * @brief Set E2E Checking result that will be used in vcc
     *
     * @param[in] e2eResult  E2E Checking result
     */
    void SetE2EResult(vrtf::com::e2exf::Result const &e2eResult) { e2eResult_ = e2eResult; }

    /**
     * @brief Get E2E checking result
     *
     * @return vrtf::com::e2exf::Result
     */
    vrtf::com::e2exf::Result GetE2EResult() const { return e2eResult_; }

    /**
     * @brief Set E2E counter that will be used by server send reply
     *
     * @param[in] e2eReceivedRequestCounter  E2E counter received by request msg
     */
    void SetE2ECounter(std::uint32_t e2eReceivedRequestCounter)
    {
        e2eReceivedRequestCounter_ = e2eReceivedRequestCounter;
    }

    /**
     * @brief Get E2E counter
     *
     * @return std::uint32_t
     */
    std::uint32_t GetE2ECounter() const { return e2eReceivedRequestCounter_; }
    /**
     * @brief Uniquely identifies the source of a type of message, as one of the identifiers of multithreaded
        scheduling tasks in the thread pool
     * @details Public base of Send request
     *
     * @return Source ID of a type of message, which composed of low and hight.
     */
    std::pair<uint64_t, uint64_t> GetMsgSourceIdToVcc() const { return msgSourceId_; }
    void SetMsgSourceIdToVcc(const std::pair<uint64_t, uint64_t> &msgSourceId) { msgSourceId_ = msgSourceId; }

protected:
    MethodMsg(const MethodMsg& other) = default;
    MethodMsg& operator=(MethodMsg const &methodMsg) = default;

private:
    EntityId entityId_;
    SessionId sessionId_;
    uint8_t *payload_;
    size_t size_;
    DriverType type_;
    InstanceId instanceId_;
    bool isTlsShakeHandsOk_ {true};
    bool isErrorMsg_ {false};
    bool trafficControl_ {false};
    vrtf::serialize::SerializeType serializeType_ = vrtf::serialize::SerializeType::SHM;
    vrtf::com::e2exf::Result e2eResult_ {vrtf::com::e2exf::Result(vrtf::com::e2exf::ProfileCheckStatus::kCheckDisabled,
        vrtf::com::e2exf::SMState::kStateMDisabled)};
    std::uint32_t e2eReceivedRequestCounter_;
    std::uint32_t e2eSourceId_ {0xFFF'FFFFU};
    std::pair<uint64_t, uint64_t> msgSourceId_ {0, 0}; // first is Hight 64 bit, sencond is low 64 bit
};
class EventMsg {
public:
    using SMState = vrtf::com::e2exf::SMState;
    using ProfileCheckStatus = vrtf::com::e2exf::ProfileCheckStatus;
    /**
     * @brief Uniquely identifies the source of a type of message, as one of the identifiers of multithreaded
        scheduling tasks in the thread pool
     * @details Public base of Send request
     *
     * @return Source ID of a type of message, which composed of low and hight.
     */
    inline std::pair<uint64_t, uint64_t> GetMsgSourceIdToVcc() const { return msgSourceId_; }
    inline void SetMsgSourceIdToVcc(const std::pair<uint64_t, uint64_t> &msgSourceId) { msgSourceId_ = msgSourceId; }
    inline void SetMbuf(Mbuf* mbuf, const uint8_t* data, uint64_t size)
    {
        mbuf_ = mbuf;
        mbufData_ = data;
        mbufSize_ = size;
    }
    inline const uint8_t* GetMbufData() const {return mbufData_;}
    inline uint64_t GetMbufDataLen() const {return mbufSize_;}
    inline void SetMbufPtr(Mbuf* mbuf) {mbuf_ = mbuf;}
    inline Mbuf* GetMbufPtr() const {return mbuf_;}
    inline void SetDataProcessFlag(bool flag) {enableDataProcess_ = flag;}
    inline bool GetDataProcessFlag() const {return enableDataProcess_;}
    inline void SetE2EResult(const vrtf::com::e2exf::Result &result) noexcept { e2eResult_ = result; }
    inline vrtf::com::e2exf::Result GetE2EResult() const noexcept { return e2eResult_; }
    inline void SetSampleId(const uint64_t &sampleId) noexcept { sampleId_ = sampleId; }
    inline std::uint64_t GetSampleId() const noexcept { return sampleId_; }
private:
    std::pair<uint64_t, uint64_t> msgSourceId_ {0, 0}; // first is Hight 64 bit, sencond is low 64 bit
    bool enableDataProcess_ = false;
    Mbuf *mbuf_ = nullptr;
    uint64_t mbufSize_ = 0U;
    const uint8_t *mbufData_ = nullptr;
    vrtf::com::e2exf::Result e2eResult_ {ProfileCheckStatus::kCheckDisabled, SMState::kStateMDisabled};
    uint64_t sampleId_ = UINT64_MAX; // for plog
};
using ServiceAvailableHandler = std::function<void(const std::vector<HandleType>&)>;
using EventHandleReceiveHandler = std::function<void(const vrtf::vcc::api::types::EventMsg&)>;
using EventReceiveHandler = std::function<void()>;
using EventSampleLostHandler = std::function<void(std::uint64_t, std::uint64_t, DriverType)>;
using MethodReceiveHandler = std::function<void(const std::shared_ptr<MethodMsg>&)>;
using SubscriberMatchedHandler = std::function<void()>;
}
}
}
}
#endif

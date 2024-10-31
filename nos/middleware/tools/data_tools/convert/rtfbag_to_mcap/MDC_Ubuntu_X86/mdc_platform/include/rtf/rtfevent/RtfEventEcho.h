/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: rtf event echo API header
 * Create: 2020-07-29
 * Notes: N/A
 */

#ifndef RTFTOOLS_RTFEVENT_ECHO_H
#define RTFTOOLS_RTFEVENT_ECHO_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ara/core/string.h"
#include "ara/core/map.h"
#include "rtf/com/rtf_com.h"
#include "rtf/internal/RtfInnerType.h"
#include "rtf/internal/tools_common_client_manager.h"
#include "rtf/maintaind/serialize/impl_type_serializationnode.h"
#include "vrtf/vcc/api/raw_buffer_for_raw_data.h"

namespace rtf {
namespace rtfevent {
struct EchoOptions {
    EchoOptions();
    ara::core::String ddsNetwork;
    ara::core::String someipNetwork;
    bool isUntypeset;
    bool isNoarr;
};
class SampleInfo;
class RtfEventEcho {
public:
    enum class EchoResult : uint8_t {
        RET_SUCCEED,
        RET_INVALID_EVENT_NAME,
        RET_INVALID_DDS_NETWORK,
        RET_INVALID_SOMEIP_NETWORK,
        RET_INIT_FAILED,
        RET_QUERY_FAILED,
        RET_SUBSCRIBE_FAILED,
        RET_NOT_SUPPORT_DATA_TYPE
    };

    using EchoCallback = std::function<void(const ara::core::String&)>;
    using EchoCallbackEx = std::function<void(SampleInfo &&)>;
    using DeserializationCallback = std::function<bool(const std::string&,
                                                        const std::vector<uint8_t>&,
                                                        std::string&)>;

    RtfEventEcho(void);
    explicit RtfEventEcho(const DeserializationCallback& deserializationCallback);
    ~RtfEventEcho(void);

    EchoResult Echo(const ara::core::String& eventName,
                    const EchoOptions& options,
                    const EchoCallback& callback) noexcept;
    EchoResult Echo(const ara::core::String& eventName, const EchoOptions& options,
        const EchoCallbackEx& callback) noexcept;
    void Stop(const ara::core::String& eventName) noexcept;

private:
    using MaintaindProxy  = rtf::maintaind::proxy::RTFMaintaindToolsServiceProxy;
    using EventInfo       = rtf::maintaind::EventRegisterInfo;
    using DDSEventInfo    = rtf::maintaind::DDSEventInfo;
    using SOMEIPEventInfo = rtf::maintaind::SomeipEventInfo;
    using ServiceList     = ara::com::ServiceHandleContainer<MaintaindProxy::HandleType>;
    using NodeHandle      = rtf::com::NodeHandle;
    using Subscriber      = rtf::com::Subscriber;
    using SerializeType   = rtf::maintaind::SerializationType;

    enum class Protocol {
        DDS,
        SOMEIP,
        UNDEFINE
    };

    struct EchoRecord {
        ara::core::String echoEventName;
        std::unique_ptr<Subscriber> subscriber = nullptr;
        EchoCallback callback = nullptr;
        EchoCallbackEx echoCallbackEx = nullptr;
        Protocol protocol = Protocol::UNDEFINE;
        bool isStop = false;
        bool isRawBuffer = false;
        bool isRecvMemory = false;
    };

    std::once_flag isInitialized_;
    std::unique_ptr<NodeHandle> nodeHandle_;
    EchoRecord echoRecord_;
    std::mutex echoMutex_;
    uint64_t receviveCount_ = 0;
    static ara::core::String applicationName_;
    DeserializationCallback deserializationCallback_;
    SerializeType serializeType_;
    uint8_t formatType_;
    bool isEnableE2E_ = false;
    bool isS2SService_ = false;
    rtf::maintaind::PduConfigInfo pduInfo_ {};
    static const uint8_t JSON_STRING = 0U;
    static const uint8_t ROS_STRING = 1U;
    static const uint8_t NO_ARR_JSON = 2U;
    static const uint8_t NO_ARR_ROS = 3U;
    std::unique_ptr<rtf::rtfevent::SampleInfo> sampleInfo_;
    std::shared_ptr<rtf::rtftools::common::ToolsCommonClientManager> toolsCommonClientManager_ = nullptr;
    ara::core::Optional<rtf::maintaind::SerializationNode> serializeRootNode_;
    bool Initialize(void);
    bool InitMaintaindConnection(void);
    bool InitializeRtfComInterface(void) noexcept;
    bool InitAndCheckValid(const ara::core::String& eventName, const EchoOptions& options, EchoResult& result);
    EchoResult CheckParam(const ara::core::String& eventName, const EchoOptions& options) const noexcept;
    std::unique_ptr<EventInfo> QueryEventInfoFromMaintaind(const ara::core::String& eventName);
    bool QueryTypeDefinitionFromMaintaind(
        ara::core::Map<ara::core::String, ara::core::String>& dataTypeJsonList) noexcept;

    EchoResult RegisterEvent(const ara::core::String& eventName,
                        const EchoOptions& options, const EventInfo& eventInfo) noexcept;
    bool RegisterDDSEvent(const ara::core::String& eventName,
                    const ara::core::String& network, const EventInfo& eventInfo) noexcept;
    bool RegisterSOMEIPEvent(const ara::core::String& eventName,
                    const ara::core::String& network, const EventInfo& eventInfo) noexcept;
    bool SubscribeEvent(const ara::core::String& eventName,
                        const EventInfo& eventInfo) noexcept;
    std::unique_ptr<Subscriber> DoSubscribe(const ara::core::String& eventName,
        const ara::core::String& dataType);
    std::unique_ptr<rtf::com::Subscriber> DoSubscribeRawBuffer(const ara::core::String& eventName,
        const ara::core::String& dataType);
    std::unique_ptr<rtf::com::Subscriber> DoSubscribeRecvMemory(const ara::core::String& eventName,
        const ara::core::String& dataType);
    void OnEventReceived(
        const ara::core::String& eventName, const ara::core::String& dataType,
        const vrtf::core::RawBuffer& data, const ara::core::Vector<rtf::OutMbufData>& mbuf = {}) noexcept;
    ara::core::String GetInstanceName(const ara::core::String& eventName) const noexcept;
    void PrintSOMEIPInfo(const ara::core::String& eventName,
                         const ara::core::String& network,
                         const rtf::stdtype::String& dataType,
                         const rtf::maintaind::SomeipEventInfo& someipEventInfo,
                         std::set<uint16_t>& eventGroupId) const;
    void SetE2ECheckResult(const ara::core::String& eventName, const rtf::com::SampleInfo& sampleInfo);
    void RtfDeserializationToGetStr(const ara::core::String& eventName, const ara::core::String& dataType,
        const vrtf::core::RawBuffer& data, const ara::core::Vector<rtf::OutMbufData>& mbuf) noexcept;
    void CustomDeserializationToGetStr(const ara::core::String& eventName, const vrtf::core::RawBuffer& data) noexcept;
    ara::core::Vector<rtf::OutMbufData> GetMbufData(const vrtf::core::RawBufferForRawData& data) const;
};
class SampleInfo {
public:
    enum class DeserializationRetCode : uint8_t {
        OK = 0U,
        UNDEFINED_ERROR,
        MISSING_MSG_INFO
    };
    SampleInfo(){}
    ~SampleInfo() = default;
    SampleInfo(const SampleInfo& other) = delete;
    SampleInfo& operator=(const SampleInfo& other) = delete;

    SampleInfo(SampleInfo && other)
        : echoResultStr_(std::move(other.echoResultStr_)),
          echoE2EResultStr_(std::move(other.echoE2EResultStr_)),
          hasDeserializationError_(other.hasDeserializationError_),
          deserializationRetCode_(std::move(other.deserializationRetCode_)),
          typeInfo_(std::move(other.typeInfo_)) {}
    SampleInfo& operator=(SampleInfo && other)
    {
        if (&other != this) {
            echoResultStr_ = std::move(other.echoResultStr_);
            echoE2EResultStr_ = std::move(other.echoE2EResultStr_);
            hasDeserializationError_ = other.hasDeserializationError_;
            deserializationRetCode_ = std::move(other.deserializationRetCode_);
            typeInfo_ = std::move(other.typeInfo_);
        }
        return *this;
    }
    ara::core::String& GetEchoResult()
    {
        return echoResultStr_;
    }
    void SetEchoResult(ara::core::String && echoResult)
    {
        echoResultStr_ = std::move(echoResult);
        hasDeserializationError_ = echoResultStr_.empty();
    }
    ara::core::String GetE2EResult()
    {
        return echoE2EResultStr_;
    }
    void SetE2EResult(ara::core::String && e2eResult)
    {
        echoE2EResultStr_ = std::move(e2eResult);
    }
    bool GetIsDeserializationError() const
    {
        return hasDeserializationError_;
    }
    DeserializationRetCode GetDeserializationRetCode() const
    {
        return deserializationRetCode_;
    }
    void SetDeserializationRetCode(DeserializationRetCode && retCode)
    {
        deserializationRetCode_ = std::move(retCode);
    }
    ara::core::String GetDeserializationRetInfo() const
    {
        return typeInfo_;
    }
    void SetDeserializationRetInfo(ara::core::String && typeInfo)
    {
        typeInfo_ = std::move(typeInfo);
    }
private:
    ara::core::String echoResultStr_;
    ara::core::String echoE2EResultStr_;
    bool hasDeserializationError_ = false;
    DeserializationRetCode deserializationRetCode_ = DeserializationRetCode::OK;
    ara::core::String typeInfo_;
};
}
}

#endif

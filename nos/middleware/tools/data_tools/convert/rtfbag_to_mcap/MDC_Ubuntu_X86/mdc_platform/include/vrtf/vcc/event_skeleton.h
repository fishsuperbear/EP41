/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: get VCC info and use to transfer driver server event mode
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_EVENTSKELETON_H
#define VRTF_VCC_EVENTSKELETON_H
#include <memory>
#include <map>
#include <type_traits>
#include "ara/core/optional.h"
#include "vrtf/vcc/utils/thread_pool.h"
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/serialize/dds_serialize.h"
#include "vrtf/vcc/serialize/someip_serialize.h"
#include "vrtf/vcc/serialize/s2s_serialize.h"
#include "vrtf/vcc/internal/traffic_crtl_policy.h"
#include "ara/com/com_error_domain.h"
#include "ara/hwcommon/log/log.h"
#include "ara/core/future.h"
#include "ara/core/promise.h"
#include "vrtf/vcc/utils/dp_adapter_handler.h"
#include "vrtf/vcc/api/raw_buffer.h"
#include "vrtf/vcc/driver/event_handler.h"
#include "vrtf/vcc/utils/event_controller.h"
#include "vrtf/vcc/utils/stats/data_statistician.h"

#include "vrtf/driver/proloc/proloc_memory_factory.h"
namespace vrtf {
namespace vcc {
/* EventHolder will be used by skeleton to initiate event operations on event */
class EventSkeleton : public std::enable_shared_from_this<EventSkeleton> {
public:
    explicit EventSkeleton(vrtf::vcc::api::types::EntityId id, bool fieldFlag = false);

    virtual ~EventSkeleton();

    virtual void SendInitData(void);

    void AddDriver(vrtf::vcc::api::types::DriverType type,
                   std::shared_ptr<vrtf::vcc::driver::EventHandler>& eventHandler);

    template<class SampleType>
    bool Send(const SampleType& data, vrtf::vcc::api::types::internal::SampleInfoImpl& info) noexcept
    {
        using namespace vrtf::vcc::api::types;
        GetLatencyTime(info);
        if (eventStats_) {
            eventStats_->IncFromUserCount();
        }
        if (trafficCtrlPolicy_ != nullptr) {
            rtf::TrafficCtrlAction action {trafficCtrlPolicy_->GetTrafficCtrlAction()};
            if (!trafficCtrlPolicy_->UpdateTrafficInfo(action)) {
                if (eventStats_ != nullptr) {
                    eventStats_->IncTrafficControlledCount();
                }
                return false;
            }
        }
        bool result {true};
        for (auto& iter : eventHandlerMap_) {
            if (firstSendFlag_) {
                /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
                /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
                RTF_DEBUG_LOG(logInstance_, (IsField() ? "Field" : "Event"), " start to send data[instanceId=",
                    GetInstanceId(), ", shortName=", GetShortNameByDriver(iter.first), ", protocol=",
                    DRIVER_TYPE_MAP.at(iter.first), "]");
                /* AXIVION enable style AutosarC++19_03-A5.0.1 */
                /* AXIVION enable style AutosarC++19_03-A5.1.1 */
                firstSendFlag_ = false;
            }
            if (!SendOut<SampleType>(iter.second, data, iter.first, info)) {
                result = false;
            } else {
                if ((eventStats_ != nullptr) && (iter.first == DriverType::DDSTYPE)) {
                    eventStats_->IncToDDSCount();
                } else if ((eventStats_ != nullptr) && (iter.first == DriverType::SOMEIPTYPE)) {
                    eventStats_->IncToSomeipCount();
                } else {
                    // Nothing to do.
                }
            }
        }
        return result;
    }

    template <typename SampleType>
    typename std::enable_if<(!vrtf::serialize::dds::IsDpRawData<SampleType>::value) &&
        (!vrtf::serialize::ros::IsRosMsg<SampleType>::VALUE), bool>::type
    SendEventByDp(std::shared_ptr<vrtf::vcc::driver::EventHandler> eventHandler, const SampleType &data,
                  vrtf::vcc::api::types::DriverType driverType,
                  vrtf::vcc::api::types::internal::SampleInfoImpl& info) noexcept
    {
        vrtf::serialize::dds::Serializer<typename std::decay<SampleType>::type> sample{
            data, eventInfo_[driverType]->GetSerializeConfig()};
        return SendCommonData(eventHandler, sample, info, driverType);
    }

    template <typename SampleType>
    typename std::enable_if<(!vrtf::serialize::dds::IsDpRawData<SampleType>::value) &&
        vrtf::serialize::ros::IsRosMsg<SampleType>::VALUE, bool>::type
    SendEventByDp(std::shared_ptr<vrtf::vcc::driver::EventHandler> eventHandler, const SampleType &data,
                  vrtf::vcc::api::types::DriverType driverType,
                  vrtf::vcc::api::types::internal::SampleInfoImpl& info) noexcept
    {
        std::shared_ptr<vrtf::vcc::RawBufferHelper> rawBufferHelper {eventInfo_[driverType]->GetRawBufferHelper()};
        if (rawBufferHelper == nullptr) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error() << "Raw buffer helper is null.";
            return false;
        }
        std::string dataType{eventInfo_[driverType]->GetDataTypeName()};
        if ((!rawBufferHelper->IsBuffMsg(dataType))) {
            vrtf::serialize::dds::Serializer<typename std::decay<SampleType>::type> sample{
                data, eventInfo_[driverType]->GetSerializeConfig()};
            return SendCommonData(eventHandler, sample, info, driverType);
        } else {
            vrtf::serialize::dds::Serializer<typename std::decay<SampleType>::type> sample{
                data, eventInfo_[driverType]->GetSerializeConfig()};
            Mbuf *mbuf = rawBufferHelper->GetMbufFromMsg(dataType,
                reinterpret_cast<void *>(const_cast<SampleType *>(&data)));
            return SendDpRawData(eventHandler, sample, mbuf, info);
        }
    }

    template <typename SampleType>
    typename std::enable_if<vrtf::serialize::dds::IsDpRawData<SampleType>::value, bool>::type
    SendEventByDp(std::shared_ptr<vrtf::vcc::driver::EventHandler> eventHandler, const SampleType &data,
                  vrtf::vcc::api::types::DriverType driverType,
                  vrtf::vcc::api::types::internal::SampleInfoImpl& info) noexcept
    {
        if (!(eventHandler->IsDpRawData())) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error() << "Failed to send raw data by dp.";
            return false;
        }
        vrtf::serialize::dds::Serializer<typename std::decay<SampleType>::type> sample{
            data, eventInfo_[driverType]->GetSerializeConfig()};
        Mbuf *mbuf {data.GetMbufPtr()};
        return SendDpRawData(eventHandler, sample, mbuf, info);
    }

    template<class SampleType>
    bool SendOut(std::shared_ptr<vrtf::vcc::driver::EventHandler> &eventHandler, const SampleType &data,
                 vrtf::vcc::api::types::DriverType driverType,
                 vrtf::vcc::api::types::internal::SampleInfoImpl& info)
    {
        using namespace vrtf::vcc::api::types;
        if (!CheckBeforeSendOut(eventHandler)) {
            return false;
        }
        if (driverType == vrtf::vcc::api::types::DriverType::PROLOCTYPE) {
            auto prolocPtr = vrtf::driver::proloc::ProlocMemoryFactory::GetEventInstance<SampleType>();
            vrtf::driver::proloc::ProlocEntityIndex prolocEntityIndex{eventInfo_[driverType]->GetServiceId(),
                                                                      eventInfo_[driverType]->GetInstanceId(),
                                                                      eventInfo_[driverType]->GetEntityId()};
            prolocPtr->StoreData(data, prolocEntityIndex, eventInfo_[driverType]->GetIsField());
            return true;
        }

        if (eventHandler->GetSerializeType() == vrtf::serialize::SerializeType::SHM) {
            if (!eventHandler->IsEnableDp()) {
                vrtf::serialize::dds::Serializer<typename std::decay<SampleType>::type> sample{
                    data, eventInfo_[driverType]->GetSerializeConfig()};
                return SendCommonData(eventHandler, sample, info, driverType);
            } else {
                bool ret {SendEventByDp<SampleType>(eventHandler, data, driverType, info)};
                if (!ret) {
                    /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                    logInstance_->error("EventSkeleton_SendOut_1", {DEFAULT_LOG_LIMIT,
                    ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                        "[EventSkeleton][Send event by dds failed]";
                    return false;
                }
            }
        } else if (eventHandler->GetSerializeType() == vrtf::serialize::SerializeType::SOMEIP) {
            vrtf::serialize::SerializationNode topSerializationNode{eventInfo_[driverType]->GetSerializationNode()};
            if (topSerializationNode.childNodeList != nullptr &&
               !topSerializationNode.childNodeList->empty()) {
                vrtf::serialize::someip::Serializer<typename std::decay<SampleType>::type> sample{
                    data, topSerializationNode.childNodeList->front(), 0, true};
                return SendCommonData(eventHandler, sample, info, driverType);
            } else {
                vrtf::serialize::someip::Serializer<typename std::decay<SampleType>::type> sample{
                    data, eventInfo_[driverType]->GetSerializeConfig()};
                return SendCommonData(eventHandler, sample, info, driverType);
            }
        } else if (eventHandler->GetSerializeType() == vrtf::serialize::SerializeType::SIGNAL_BASED) {
            return SendSignalData(eventHandler, data,driverType, info);
        } else {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->debug() << "Wrong serialize type, deserialize error!";
            return false;
        }
        return true;
    }

    /**
     * @brief Allocate rawBuffer
     * @details Allocate rawBuffer
     *
     * @param size the size will be allocated
     * @return RawBuffer return one RawBuffer for pub
     *   @retval rawBuffer allocate buffer successful
     *   @retval nullptr allocate buffer failed
     */
    vrtf::vcc::api::types::RawBuffer AllocateRawBuffer(const size_t& size);

    /**
     * @brief pub rawBuffer
     * @details pub rawBuffer
     *
     * @param rawBuffer the rawBuffer class store data pointer to be send
     */
    void DeAllocateRawBuffer(api::types::RawBuffer& rawBuffer);

    /**
     * @brief pub rawBuffer
     * @details pub rawBuffer
     *
     * @param rawBuffer the rawBuffer class store data pointer to be send
     */
    bool PubRawBuffer(api::types::RawBuffer& rawBuffer,
                      vrtf::vcc::api::types::internal::SampleInfoImpl& info);

    bool IsField(void) const;

    void SetShortNameByDriver(vrtf::vcc::api::types::DriverType const &drivertype, vrtf::core::String const &shortname);

    vrtf::core::String GetShortNameByDriver(vrtf::vcc::api::types::DriverType const &drivertype);

    vrtf::vcc::api::types::InstanceId GetInstanceId() const;

    void SetInstanceId(vrtf::vcc::api::types::InstanceId const &instanceId);

    void SetTrafficCtrl(const std::shared_ptr<rtf::TrafficCtrlPolicy>& policy);

    void SetEventInfo(const std::map<vrtf::vcc::api::types::DriverType,
                      std::shared_ptr<vrtf::vcc::api::types::EventInfo>>& eventInfo);
    /**
     * @brief Set this proxy latency analysis mode
     * @details Set this proxy latency analysis mode
     * @param[in] mode latency analysis mode
     */
    void SetLatencyAnalysisMode(const utils::LatencyAnalysisMode& mode);
    void EnableDataStatistics(vrtf::vcc::utils::stats::DataStatistician::PeriodType const period);
    api::types::ReturnCode WaitForFlush(const std::uint32_t waitMs) noexcept;
protected:
    vrtf::vcc::api::types::EntityId const id_;
    bool fieldFlag_ {false};
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    bool firstSendFlag_ {true};
    std::map<vrtf::vcc::api::types::DriverType, vrtf::core::String> shortName_;
    vrtf::vcc::api::types::InstanceId instanceId_ {vrtf::vcc::api::types::UNDEFINED_INSTANCEID};
private:
    template<class Serializer>
    bool SendCommonData(std::shared_ptr<vrtf::vcc::driver::EventHandler> &eventHandler, Serializer &serializer,
                        vrtf::vcc::api::types::internal::SampleInfoImpl& info,
                        vrtf::vcc::api::types::DriverType const &type) noexcept
    {
        using namespace vrtf::vcc::api::types;
        size_t size = serializer.GetSize();
        if (size >= (vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE - utils::TLV_TIME_TOTAL_SIZE)) {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->error("EventSkeleton_SendCommonData",
            {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) << "[EventSkeleton]" <<
                "[Serialize failed, maybe length filed is less than actual size][instanceId=" << GetInstanceId() <<
                ", shortName=" << GetShortNameByDriver(type) << ", protocol=" << DRIVER_TYPE_MAP.at(type) << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return false;
        }
        std::uint8_t* buffer {nullptr};
        if ((delayMode_ == utils::LatencyAnalysisMode::ENABLE) && (info.isEvent_) &&
            CheckTimeIsValid(info.sendTime_)) {
            buffer = eventHandler->AllocateBuffer(static_cast<uint32_t>(size + utils::TLV_TIME_TOTAL_SIZE));
            if (buffer != nullptr) {
                utils::TlvHelper::AddTlvTimeStamp(buffer + size, info.sendTime_);
                size += utils::TLV_TIME_TOTAL_SIZE;
            }
        } else {
            buffer = eventHandler->AllocateBuffer(static_cast<uint32_t>(size));
        }
        if (!CheckCommonDataBuffer(buffer, info, type)) {
            return false;
        }
        serializer.Serialize(buffer);
        if (info.plogInfo_ != nullptr) {
            info.plogInfo_->WriteTimeStamp(
                utils::PlogServerTimeStampNode::SERIALIZE_DATA, TO_PLOG_DRIVER_TYPE_MAP.at(type));
        }
        bool ret {eventHandler->SendEvent(buffer, size, info)};
        /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
        RTF_DEBUG_LOG(logInstance_, "Data size is ", size, " in entity ", id_);
        return ret;
    }

    template<class SampleType>
    bool SendSignalData(const std::shared_ptr<vrtf::vcc::driver::EventHandler>& eventHandler, const SampleType &data,
        vrtf::vcc::api::types::DriverType driverType,
        vrtf::vcc::api::types::internal::SampleInfoImpl &info) noexcept
    {
        using namespace vrtf::vcc::api::types;
        const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu{eventInfo_[driverType]->GetPdu()};
        if (pdu == nullptr) {
            RTF_ERROR_LOG_SPR(logInstance_, "EventSkeleton_SendSignal_1", DEFAULT_LOG_LIMIT,
                "[EventSkeleton][Send event fail with null pdu]");
            return false;
        }
        vrtf::serialize::s2s::Serializer<typename std::decay<SampleType>::type> serializer{data, pdu};
        if (!eventDataCount_.has_value()) {
            eventDataCount_ = serializer.GetSignalCount();
        }
        if (pdu->ISignalCount() != eventDataCount_) {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            RTF_ERROR_LOG_SPR(logInstance_, "EventSkeleton_SendSignal_2", DEFAULT_LOG_LIMIT, "[EventSkeleton]",
                "[Signal-based event data is not matched with the pdu][entityId=", id_, ", eventDataCount=",
                eventDataCount_.value_or(USHRT_MAX), ", pduSignalCount=", pdu->ISignalCount(), "]");
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return false;
        }
        std::size_t size = pdu->LengthWithoutDynamic() + serializer.GetDynamicLength();
        std::uint8_t* buffer {nullptr};
        buffer = eventHandler->AllocateBuffer(static_cast<uint32_t>(size));
        if (!CheckCommonDataBuffer(buffer, info, driverType)) {
            return false;
        }
        serializer.Serialize(buffer);
        bool ret {eventHandler->SendEvent(buffer, size, info)};
        RTF_DEBUG_LOG(logInstance_, "Data size is ", size, " in entity ", id_);
        return ret;
    }

    bool CheckBeforeSendOut(std::shared_ptr<vrtf::vcc::driver::EventHandler> const &eventHandler);

    template<class Serializer>
    bool SendDpRawData(std::shared_ptr<vrtf::vcc::driver::EventHandler> &eventHandler,
        Serializer &serializer, Mbuf *mbuf,
        vrtf::vcc::api::types::internal::SampleInfoImpl& info) noexcept
    {
        uint8_t *buffer {nullptr};
        size_t size {serializer.GetSize()};
        if (!CheckBeforeSendDpRawData(mbuf, size)) {
            return false;
        }
        std::uint8_t latencySize {0U};
        if ((delayMode_ == utils::LatencyAnalysisMode::ENABLE) && (info.isEvent_) &&
            CheckTimeIsValid(info.sendTime_)) {
            latencySize = utils::TLV_TIME_TOTAL_SIZE;
        }
        if ((eventHandler->GetE2EHeaderSize() != vrtf::com::e2exf::UNDEFINED_HEADER_SIZE) && (latencySize > 0)) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error("EventSkeleton_SendDpRawData_3", {vrtf::vcc::api::types::DEFAULT_LOG_LIMIT,
            ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[EventSkeleton][Using latency and e2e in the same time which is not supported]";
            return false;
        }
        buffer = GetRawDataSerializeBuffer(eventHandler, mbuf, size + latencySize);
        if (buffer == nullptr) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error("EventSkeleton_SendDpRawData_4", {vrtf::vcc::api::types::DEFAULT_LOG_LIMIT,
            ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[EventSkeleton][Failed to get raw data serialize buffer]";
            if (eventStats_) {
                eventStats_->IncRawDataGetSerializeBufferFailedCount();
            }
            static_cast<void>(dpHandler_->MbufFree(mbuf));
            return false;
        } else { // The format in private area is [Private data length][E2E Header][Private data][Latency time stamp]
            if ((latencySize > 0) && (info.isEvent_)) { // Write latency time stamp
                utils::TlvHelper::AddTlvTimeStamp(buffer + size, info.sendTime_);
            }
        }
        serializer.Serialize(buffer);
        if (info.plogInfo_ != nullptr) {
            info.plogInfo_->WriteTimeStamp(
                utils::PlogServerTimeStampNode::SERIALIZE_DATA, utils::PlogDriverType::DDS);
        }
        bool ret {eventHandler->SendEvent(reinterpret_cast<uint8_t *>(mbuf), size, info)};
        /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
        /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
        logInstance_->debug() << "Raw Data private size is " << size << ", and the latency size is " <<
                                 latencySize << " in entity " << id_;
        /* AXIVION enable style AutosarC++19_03-A5.0.1 */
        /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        return ret;
    }

    uint8_t* GetRawDataSerializeBuffer(const std::shared_ptr<vrtf::vcc::driver::EventHandler> &eventHandler,
        Mbuf * const mbuf, size_t size);

    void RecordStatistics();
    std::string MakePlogStatisticsStr() noexcept;

    inline bool TrafficProc(api::types::RawBuffer& rawBuffer);
    void GetLatencyTime(vrtf::vcc::api::types::internal::SampleInfoImpl& info) const noexcept;
    bool CheckTimeIsValid(const timespec& time) const noexcept;

    bool CheckCommonDataBuffer(const uint8_t* buffer,
                               vrtf::vcc::api::types::internal::SampleInfoImpl const &info,
                               vrtf::vcc::api::types::DriverType const &type);

    bool CheckBeforeSendDpRawData(Mbuf *mbuf, size_t size) const
    {
        if (dpHandler_ == nullptr) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error("EventSkeleton_SendDpRawData_1", {vrtf::vcc::api::types::DEFAULT_LOG_LIMIT,
            ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[EventSkeleton][DpAdapterHandler is null]";
            return false;
        }
        if (size > dpHandler_->GetAvailabeLenth()) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error("EventSkeleton_SendDpRawData_2", {vrtf::vcc::api::types::DEFAULT_LOG_LIMIT,
            ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[EventSkeleton][Too many private data for mbuf raw data][maxAvailableSize="
                << dpHandler_->GetAvailabeLenth() << ", size=" << size << "]";
            if (eventStats_) {
                eventStats_->IncRawDataGetAvailabeLenthFailedCount();
            }
            static_cast<void>(dpHandler_->MbufFree(mbuf));
            return false;
        }
        return true;
    }
    using EventHandlerMap = std::map<vcc::api::types::DriverType, std::shared_ptr<vcc::driver::EventHandler>>;
    bool CheckValidForRawMemory(const EventHandlerMap &eventHandlerMap) const;
    EventHandlerMap eventHandlerMap_;
    std::shared_ptr<vrtf::vcc::utils::DpAdapterHandler> dpHandler_ {nullptr};
    std::shared_ptr<rtf::TrafficCtrlPolicy> trafficCtrlPolicy_;
    std::map<vrtf::vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::EventInfo>> eventInfo_;
    const std::map<vrtf::vcc::api::types::DriverType, vrtf::vcc::utils::PlogDriverType> TO_PLOG_DRIVER_TYPE_MAP = {
        { vrtf::vcc::api::types::DriverType::DDSTYPE, vrtf::vcc::utils::PlogDriverType::DDS },
        { vrtf::vcc::api::types::DriverType::SOMEIPTYPE, vrtf::vcc::utils::PlogDriverType::SOMEIP }
    };
    utils::LatencyAnalysisMode delayMode_ {utils::LatencyAnalysisMode::DISABLE};
    std::shared_ptr<vrtf::vcc::utils::EventController> ctrler_ {nullptr};
    std::shared_ptr<vrtf::vcc::utils::stats::DataStatistician> dataStatistian_ {nullptr};
    std::shared_ptr<utils::stats::EventSkeletonStats> eventStats_ {nullptr};
    vrtf::vcc::utils::EventHandle statisticHandle_ {vrtf::vcc::utils::UNDEFINED_EVENT_UID};
    bool validForRawMemory_ = true;
    ara::core::Optional<uint16_t> eventDataCount_;
};

template<class T>
class FieldSkelton : public EventSkeleton {
public:
    FieldSkelton(vrtf::vcc::api::types::EntityId id, bool hasNotify)
        : EventSkeleton(id, true), isInitialized_(false), hasNotify_(hasNotify)
    {
        using namespace ara::godel::common;
        logInstance_ = log::Log::GetLog("CM");
    }
    ~FieldSkelton() override = default;
    /**
     * @brief field send interface
     * @details if use before offerservice, update initdata and store
     *          if use after offerservice, the data will be send by this interface
     *
     * @param[in] data the data which will be sent
     * @param[in] isOfferService if isOfferService is true represent have offerservice
     *            if isOfferService is false represent have not offerservice
     */
    template <typename U = T>
    typename std::enable_if<(!vrtf::serialize::ros::IsRosMsg<U>::VALUE) &&
                            (!vrtf::serialize::ros::IsRosBuiltinMsg<T>::VALUE) &&
                            (!vrtf::serialize::ros::IsShapeShifterMsg<T>::VALUE), void>::type
    Send(const typename std::decay<T>::type& data, bool isOfferService,
         vrtf::vcc::api::types::internal::SampleInfoImpl& info)
    {
        isInitialized_ = true;
        if (!isOfferService) {
            field_ = data;
            fieldInitValue_ = data;
            return;
        }

        if (!hasNotify_) {
            field_ = data;
            return;
        }

        {
            std::lock_guard<std::mutex> guard{firstSendMutex_};
            if (firstSend_) {
                firstSend_ = false;
                field_ = data;
                EventSkeleton::Send<typename std::decay<T>::type>(data, info);
                return;
            }
        }
        if (data == field_) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->verbose()<< "Update the same notifier value";
        } else {
            field_ = data;
            EventSkeleton::Send<typename std::decay<T>::type>(data, info);
        }
    }

    template <typename U = T>
    typename std::enable_if<vrtf::serialize::ros::IsRosMsg<U>::VALUE ||
                            vrtf::serialize::ros::IsRosBuiltinMsg<T>::VALUE ||
                            vrtf::serialize::ros::IsShapeShifterMsg<T>::VALUE, void>::type
    Send(const typename std::decay<T>::type& data, bool isOfferService,
         vrtf::vcc::api::types::internal::SampleInfoImpl& info)
    {
        static_cast<void>(data);
        static_cast<void>(isOfferService);
        static_cast<void>(info);
    }

    void SendInitData(void) override
    {
        /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
        logInstance_->verbose()<< "New subscriber online for field " << id_;
        if (hasNotify_) {
            api::types::internal::SampleInfoImpl sampleInfo;
            EventSkeleton::Send<T>(fieldInitValue_, sampleInfo);
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->verbose()<< "Send field value to the new subscriber.";
        }
    }

    void RegisterGetMethod(std::function<vrtf::core::Future<T>(void)> getHandler)
    {
        getHandler_ = getHandler;
    }

    void RegisterSetMethod(std::function<vrtf::core::Future<T>(const T& value)> setHandler)
    {
        setHandler_ = setHandler;
    }

    /**
     * @brief query the getHandler
     * @return getHandler or nullptr if not registered.
     */
    const std::function<vrtf::core::Future<T>()>& QueryGetMethod()
    {
        return getHandler_;
    }

    /**
     * @brief query the setHandler
     * @return setHandler or nullptr if not registered.
     */
    const std::function<vrtf::core::Future<T>(const T& value)>& QuerySetMethod()
    {
        return setHandler_;
    }

    /**
     * @brief check if the field initialized.
     * @details the initial data is set by calling Update() before OfferService() by APP.
     * @return the result of check
     *   @retval true the field has initialized
     *   @retval false the field has not initialized
     */
    bool HasInitData()
    {
        return isInitialized_;
    }

    void SetFieldStatus(bool isInit)
    {
        isInitialized_ = isInit;
    }

    vrtf::core::Future<T> UpdateField(const T& data, bool isDeserializefail = false)
    {
        vrtf::core::Promise<T> promise;
        vrtf::core::Future<T> future {promise.get_future()};
        if (isDeserializefail) {
            if (!isInitialized_) {
                promise.SetError(vrtf::core::ErrorCode(vrtf::vcc::api::types::ComErrc::kNetworkBindingFailure));
            } else {
                promise.set_value(fieldInitValue_);
            }
            return future;
        }
        if (setHandler_ != nullptr) {
            auto userFutureData = setHandler_(data).get();
            api::types::internal::SampleInfoImpl sampleInfo;
            Send(userFutureData, true, sampleInfo);
            promise.set_value(userFutureData);
        } else {
            typename std::decay<T>::type interData;
            // Fix me: error code should be return
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->warn() << "Unregisterd set handler is called with entityId " << id_;
            promise.set_value(interData);
        }
        return future;
    }

    virtual vrtf::core::Future<T> GetField(void)
    {
        vrtf::core::Promise<T> promise;
        vrtf::core::Future<T> future {promise.get_future()};
        if (getHandler_ != nullptr) {
            auto userFuture = getHandler_();
            auto userFutureData = userFuture.get();
            promise.set_value(userFutureData);
        } else {
            // Fix me: error code should be return
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->warn() << "Unregisterd get handler is called with entityId " << id_;
            promise.set_value(field_);
        }
        return future;
    }

private:
    // UpdateField should be called when the obj is created
    T field_;
    T fieldInitValue_;
    bool isInitialized_ {false};
    bool hasNotify_ {false};
    bool firstSend_ {true};
    std::mutex firstSendMutex_;
    std::function<vrtf::core::Future<T>(void)> getHandler_ {nullptr};
    std::function<vrtf::core::Future<T>(const T& value)> setHandler_ {nullptr};
};
}
}
#endif

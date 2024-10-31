/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: skeleton in vcc
 * Create: 2019-11-19
 */
#ifndef INC_ARA_GODEL_COMMON_VCC_SKELETON_HPP
#define INC_ARA_GODEL_COMMON_VCC_SKELETON_HPP

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <functional>
#include "ara/hwcommon/log/log.h"
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/utils/safe_map.h"
#include "vrtf/vcc/vcc.h"
#include "vrtf/vcc/internal/traffic_crtl_policy.h"
namespace ara {
namespace com {
class ThreadGroup;
}
}
namespace vrtf {
namespace vcc {
class Skeleton {
public:
    Skeleton(const std::string& serviceName, const vrtf::vcc::api::types::MethodCallProcessingMode& methodMode);
    Skeleton(const std::string& serviceName, const vrtf::vcc::api::types::MethodCallProcessingMode& methodMode,
             const bool& discoveyMode);
    Skeleton(const Skeleton &) = delete;
    Skeleton& operator=(const Skeleton &) = delete;
    Skeleton(Skeleton && other) = default;
    Skeleton& operator=(Skeleton && other) = default;
    ~Skeleton();
    bool SetEventTrafficCtrl(const std::shared_ptr<rtf::TrafficCtrlPolicy>& policy,
        const vrtf::vcc::api::types::EntityId& id);
    bool SetMethodTrafficCtrl(const std::shared_ptr<rtf::TrafficCtrlPolicy>& policy,
        const vrtf::vcc::api::types::EntityId& id);
    bool SetMethodThreadGroup(const std::shared_ptr<ara::com::ThreadGroup>& threadGroup,
        const vrtf::vcc::api::types::EntityId& id);
    void OfferService(const std::map<vrtf::vcc::api::types::DriverType,
                      std::shared_ptr<vrtf::vcc::api::types::ServiceDiscoveryInfo>>& protocolData);
    /**
     * @brief Allocate rawBuffer
     * @details Allocate rawBuffer
     *
     * @param size the size will be allocated
     * @param id EntityId
     * @return RawBuffer return one RawBuffer for pub
     *   @retval rawBuffer allocate buffer successful
     *   @retval nullptr allocate buffer failed
     */
    vrtf::vcc::api::types::RawBuffer AllocateRawBuffer(const size_t& size, const vrtf::vcc::api::types::EntityId& id);
    /**
     * @brief deallocate buffer when rawBuffer is unuseless
     * @details deallocate buffer when rawBuffer is unuseless
     *
     * @param rawBuffer the rawBuffer class store data pointer to be destroy
     * @param id EntityId
     */
    void DeAllocateRawBuffer(api::types::RawBuffer& rawBuffer, const vrtf::vcc::api::types::EntityId& id);
    /**
     * @brief pub rawBuffer
     * @details pub rawBuffer
     *
     * @param rawBuffer the rawBuffer class store data pointer to be send
     * @param id EntityId
     * @param info use to store and print time info to plog module
     */
    vrtf::core::Result<void> PubRawBuffer(api::types::RawBuffer& rawBuffer, const vrtf::vcc::api::types::EntityId& id,
        vrtf::vcc::api::types::internal::SampleInfoImpl& info);
    /**
     * @brief skeleton Initialize Event
     * @details invoke vcc to initializeEventSkel
     *
     * @param protocolData the event Info read from config file
     * @return Event is init successful
     *   @retval true Event Init is successful
     *   @retval false Event Init is fail
     * @note AUTOSAR AP R19-11 RS_CM_00201
     */
    bool InitializeEvent(const std::map<vrtf::vcc::api::types::DriverType,
                         std::shared_ptr<vrtf::vcc::api::types::EventInfo>>& protocolData);
    std::shared_ptr<EventSkeleton> GetEventSkeleton(api::types::EntityId id);
    void StopOfferService(
        const std::map<vrtf::vcc::api::types::DriverType,
        std::shared_ptr<vrtf::vcc::api::types::ServiceDiscoveryInfo>>& protocolData) noexcept;
    api::types::ReturnCode WaitForFlush(const std::uint32_t waitMs, const vrtf::vcc::api::types::EntityId id) noexcept;
    template<class SampleType>
    std::unique_ptr<SampleType> Allocate() const
    {
        return std::make_unique<SampleType>();
    }

    template<class SampleType>
    ara::core::Result<void> Send(const SampleType& data, vrtf::vcc::api::types::EntityId id,
                                 vrtf::vcc::api::types::internal::SampleInfoImpl& info)
    {
        return vcc_->Send<SampleType>(data, id, info);
    }

    /**
     * @brief skeleton Initialize Method
     * @details invoke vcc to initializeMethod
     *
     * @param protocolData the method Info read from config file
     * @return method is init successful
     *   @retval true method Init is successful
     *   @retval false method Init is fail
     * @note AUTOSAR AP R19-11 RS_CM_00211
     */
    template<class Result>
    bool InitializeMethod(const std::map<vrtf::vcc::api::types::DriverType,
        std::shared_ptr<vrtf::vcc::api::types::MethodInfo>>& protocolData)
    {
        return vcc_->InitializeMethodSkel<Result>(protocolData);
    }

    template<class T, class Result, class... Args>
    void RegisterMethod(Result(T::*callback)(Args...), T& c, const vrtf::vcc::api::types::EntityId& id,
        const vrtf::vcc::api::types::ThreadPoolPair& tPair = {vrtf::vcc::api::types::ThreadPoolType::ARA_COM, nullptr})
    {
        vcc_->RegisterMethod<T, Result, Args...>(callback, c, id, tPair);
    }

    template<class Result>
    void RegisterMethod(const vrtf::vcc::api::types::EntityId& id, std::function<Result()> callback,
        const vrtf::vcc::api::types::ThreadPoolPair& tPair = {vrtf::vcc::api::types::ThreadPoolType::ARA_COM, nullptr})
    {
        vcc_->RegisterMethod<Result>(id, callback, tPair);
    }

    template<class Result, class... Args>
    void RegisterMethod(const vrtf::vcc::api::types::EntityId& id, std::function<Result(Args...)> callback,
        const vrtf::vcc::api::types::ThreadPoolPair& tPair = {vrtf::vcc::api::types::ThreadPoolType::ARA_COM, nullptr})
    {
        vcc_->RegisterMethod<Result, Args...>(id, callback, tPair);
    }

    template<class Result, class... Args>
    void RegisterMethod(const vrtf::vcc::api::types::EntityId& id,
        std::function<Result(vrtf::com::e2exf::Result, Args...)> callback,
        const vrtf::vcc::api::types::ThreadPoolPair& tPair = {vrtf::vcc::api::types::ThreadPoolType::ARA_COM, nullptr})
    {
        vcc_->RegisterMethod<Result, Args...>(id, callback, tPair);
    }

    template<class SampleType>
    void RegisterGetMethod(std::function<vrtf::core::Future<SampleType>()> getHandler,
                           const vrtf::vcc::api::types::EntityId& id)
    {
        vcc_->RegisterGetMethod<SampleType>(getHandler, id);
    }

    template<class SampleType>
    void RegisterSetMethod(std::function<vrtf::core::Future<SampleType>(const SampleType& value)> setHandler,
                           const vrtf::vcc::api::types::EntityId& id)
    {
        vcc_->RegisterSetMethod<SampleType>(setHandler, id);
    }

    /**
     * @brief Query field info of the field
     * @details Field info: hasGetter_, hasSetter_, hasNotifier_
     * @param[in] id EntityId for the field
     * @param[out] info Field info for the field parsed from config file.
     */
    void QueryFieldInfo(const vrtf::vcc::api::types::EntityId& id,
        std::shared_ptr<vrtf::vcc::api::types::FieldInfo>& info);

    /**
     * @brief Set field to uninitialized state
     * @param[in] id method entity id.
     */
    template<class SampleType>
    void ResetInitState(const vrtf::vcc::api::types::EntityId& id) const
    {
        vcc_->ResetInitState<SampleType>(id);
    }

    /**
     * @brief check if the field initialized.
     * @return the result of check
     *   @retval true the field has initialized
     *   @retval false the field has not initialized
     */
    template<class SampleType>
    bool IsFieldInitialized(const vrtf::vcc::api::types::EntityId& id)
    {
        return vcc_->IsFieldInitialized<SampleType>(id);
    }

    /**
     * @brief Query sethandler of the field
     * @param[in] id EntityId for the field
     * @param[out] handler getHandler for the field registered by RegisterGetMethod().
     *
     */
    template<class SampleType>
    void QueryGetMethod(const vrtf::vcc::api::types::EntityId& id,
        std::function<vrtf::core::Future<SampleType>()>& handler)
    {
        vcc_->QueryGetMethod<SampleType>(id, handler);
    }

    /**
     * @brief Query sethandler of the field
     * @param[in] id-EntityId for the field
     * @param[out] handler- setHandler for the field registered by RegisterSetMethod().
     *
     */
    template<class SampleType>
    void QuerySetMethod(const vrtf::vcc::api::types::EntityId& id,
        std::function<vrtf::core::Future<SampleType>(const SampleType& value)>& handler)
    {
        vcc_->QuerySetMethod<SampleType>(id, handler);
    }

    bool SetMethodThreadNumber(const std::uint16_t& number, const std::uint16_t& queueSize);
    void UnregisterAllMethod();
    bool ProcessNextMethodCall();
    /**
     * @brief skeleton Initialize Field
     * @details distingush field has notify/get/set and invoke vcc to initializeField
     *
     * @param fieldData the field Info read from config file
     * @return field is init successful
     *   @retval true field Init is successful
     *   @retval false field Init is fail
     * @note AUTOSAR AP R19-11 RS_CM_00216 RS_CM_00217 RS_CM_00218
     */
    template<class SampleType>
    bool InitializeField(std::map<vcc::api::types::DriverType,
        std::shared_ptr<vrtf::vcc::api::types::FieldInfo>> fieldData)
    {
        FieldInfo fieldInfo{ExtractFieldInfo(fieldData)};
        vcc_->SetFieldInfo(fieldData, fieldInfo.eventid_);
        if (vcc_->InitializeFieldSkel<SampleType>(fieldInfo.eventData_, fieldInfo.hasNotifier_) == false) {
            return false;
        }
        std::shared_ptr<vrtf::vcc::FieldSkelton<SampleType>> eventSkel =
            std::static_pointer_cast<vrtf::vcc::FieldSkelton<SampleType>>(vcc_->GetEventSkel(fieldInfo.eventid_));
        if (eventSkel == nullptr) {
            logInstance_->error()<< "[Skeleton][Fail to init field][id=" << fieldInfo.eventid_ << "]";
            return false;
        }

        if (fieldInfo.hasGetter_) {
            if ((vcc_->InitializeMethodSkel<ara::core::Future<SampleType>>)(fieldInfo.getMethodData_) == false) {
                return false;
            }
            vcc_->RegisterMethod(&vrtf::vcc::FieldSkelton<SampleType>::GetField, *eventSkel, fieldInfo.getterid_);
        }

        if (fieldInfo.hasSetter_) {
            if ((vcc_->InitializeMethodSkel<ara::core::Future<SampleType>>)(fieldInfo.setMethodData_) == false) {
                return false;
            }
            vcc_->RegisterMethodForSet(&vrtf::vcc::FieldSkelton<SampleType>::UpdateField,
                *eventSkel, fieldInfo.setterid_);
        }
        return true;
    }

// for server
/**
 * @brief Register field set/get handler before OfferService
 * @details This function is use to register get/set method before OfferService
 * @param[in] info this params stores field info which read from config file
 */
    template<class SampleType>
    void RegisterFieldHandler(const std::shared_ptr<vrtf::vcc::api::types::FieldInfo>& info)
    {
        auto eventId = info->GetEventInfo()->GetEntityId();
        std::shared_ptr<vrtf::vcc::FieldSkelton<SampleType>> eventSkel =
            std::static_pointer_cast<vrtf::vcc::FieldSkelton<SampleType>>(vcc_->GetEventSkel(eventId));
        if (eventSkel == nullptr) {
            logInstance_->error()<< "[Skeleton][Fail to register field handler][id=" << eventId << "]";
            return;
        }
        if (info->IsHasGetter()) {
            vcc_->RegisterMethod(&vrtf::vcc::FieldSkelton<SampleType>::GetField, *eventSkel,
                info->GetGetterMethodInfo()->GetEntityId());
        }
        if (info->IsHasSetter()) {
            vcc_->RegisterMethodForSet(&vrtf::vcc::FieldSkelton<SampleType>::UpdateField, *eventSkel,
                info->GetSetterMethodInfo()->GetEntityId());
        }
    }
    /**
     * @brief   Register E2E error handler which will be called if the E2E checking result is error
     *
     * @param[in] callback   The handler of handling E2E checking result is error
     * @param[in] instance      The class of callback belong to
     * @param[in] id         EntityId
     */
    template<class T>
    void RegisterE2EErrorHandler(void(T::*callback)(vrtf::vcc::api::types::E2EErrorCode,
        vrtf::vcc::api::types::E2EDataId, vrtf::vcc::api::types::MessageCounter), T& instance)
    {
        vcc_->RegisterE2EErrorHandler([callback, &instance](vrtf::vcc::api::types::E2EErrorCode errorCode,
        vrtf::vcc::api::types::E2EDataId dataId, vrtf::vcc::api::types::MessageCounter messageCounter) {
            (instance.*callback)(errorCode, dataId, messageCounter);
        });
    }

    void EnableDataStatistics(const vrtf::vcc::utils::stats::DataStatistician::PeriodType period);
    void UnEnableDataStatistics();
private:
    struct FieldInfo {
        std::map<vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::EventInfo>> eventData_;
        std::map<vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::MethodInfo>> getMethodData_;
        std::map<vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::MethodInfo>> setMethodData_;
        vcc::api::types::EntityId eventid_;
        vcc::api::types::EntityId getterid_;
        vcc::api::types::EntityId setterid_;
        bool hasGetter_ = false;
        bool hasSetter_ = false;
        bool hasNotifier_ = false;
    };
    FieldInfo ExtractFieldInfo(std::map<vcc::api::types::DriverType,
        std::shared_ptr<vrtf::vcc::api::types::FieldInfo>> fieldInfo) const;
    std::string driverName;
    std::shared_ptr<Vcc> vcc_;
    std::shared_ptr<vrtf::vcc::utils::ThreadPool> pool_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
};
}
}

#endif /* INC_ARA_GODEL_COMMON_VCC_SKELETON_HPP_ */

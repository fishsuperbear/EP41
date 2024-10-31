/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Adapter layer between Ros and Vcc Proxy
 * Create: 2020-04-22
 */
#ifndef RTF_COM_ADAPTER_ROS_SKELETON_ADAPTER_H
#define RTF_COM_ADAPTER_ROS_SKELETON_ADAPTER_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <ara/core/promise.h>

#include "rtf/com/types/ros_types.h"
#include "rtf/com/entity/thread_group.h"
#include "rtf/com/utils/logger.h"
#include "vrtf/vcc/api/raw_buffer.h"
#include "vrtf/vcc/utils/rtf_spin_lock.h"
#include "rtf/com/types/method_result.h"
#include "vrtf/vcc/api/vcc_method_return_type.h"
#include "rtf/com/utils/proloc_manager.h"
namespace vrtf {
namespace driver {
namespace someip {
class SomeipInterface;
}
}
}
namespace rtf {
namespace com {
namespace utils {
    class SomeipJsonHelper;
}
namespace adapter {
class RosSkeletonAdapter : public std::enable_shared_from_this<RosSkeletonAdapter> {
public:
    /**
     * @brief RosSkeletonAdapter constructor
     */
    RosSkeletonAdapter(void);

    /**
     * @brief RosSkeletonAdapter destructor
     */
    ~RosSkeletonAdapter(void);

    /**
     * @brief Initialize RosSkeletonAdapter
     * @param[in] entityAttr     The entity attribute
     * @param[in] maintainConfig The config for maintain
     * @param[in] threadPool     The thread poll that handles operations
     */
    bool Initialize(const EntityAttr& entityAttr, const MaintainConfig& maintainConfig,
                    const std::shared_ptr<VccThreadPool>& threadPool) noexcept;

    /**
     * @brief Return whether the adapter is initialized
     * @return Is the adapter initialized
     */
    bool IsInitialized(void) const noexcept;

    /**
     * @brief Return whether the adapter is ready to send/receive data
     * @return Is the adapter ready
     */
    bool IsValid(void) const noexcept;

    /**
     * @brief Create and register a method
     *
     * @tparam Request the type of request message
     * @tparam Response  the type of reponse message
     * @param[in] callback  the callback function
     * @return bool the result of create and register a method
     */
    template<class Request, class Response>
    bool RegisterMethod(std::function<bool(Request&, Response&)> callback) noexcept
    {
        using namespace rtf::com::utils;

        // Initialize Method
        /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
        RTF_DEBUG_LOG(logger_, "[RTFCOM] Registering method '", uri_, "'...");
        if (!InitializeMethod<vrtf::core::Future<Response>>()) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
            RTF_ERROR_LOG(logger_, "[RTFCOM] Failed to register method '", uri_, "'");
            return false;
        }
        std::function<vrtf::core::Future<Response>(Request)> methodHandler =
            [callback] (Request request) -> vrtf::core::Future<Response> {
                Response response;
                callback(request, response);
                vrtf::core::Promise<Response> promise;
                promise.set_value(response);
                return promise.get_future();
            };
        std::unordered_map<AdapterProtocol, std::shared_ptr<Entity>> entityMapBack;
        {
            std::lock_guard<std::mutex> lock{entityMapMutex_};
            entityMapBack = entityMap_;
        }
        for (auto& entityMapIterator : entityMapBack) {
            const auto& protocol = entityMapIterator.first;
            const auto& entity   = entityMapIterator.second;
            const auto& skeleton = entity->skeleton;
            // Register method
            VccThreadPoolPair poolPair {VccThreadPoolType::RTF_COM, threadPool_};
            skeleton->RegisterMethod(entityId_, methodHandler, poolPair);
            // Offer entity
            if (!isUsingDefaultConfig_) {
                OfferEntity(entity, protocol);
            } else {
                if (vrtf::vcc::utils::stats::DataStatistician::GetInstance()->IsEnable()) {
                    skeleton->EnableDataStatistics(vrtf::vcc::utils::stats::DataStatistician::GetInstance()->Period());
                }
            }
        }
        /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
        RTF_DEBUG_LOG(logger_, "[RTFCOM] Method '", uri_, "' is registered");
        return true;
    }

    /**
     * @brief Create and register a method
     *
     * @tparam Request the type of request message
     * @tparam Response  the type of reponse message
     * @param callback  the callback function with E2E result
     * @return bool the result of create and register a method
     */
    template<class Request, class Response>
    bool RegisterMethod(std::function<bool(Request&, Response&, MethodServerResult&)> callback) noexcept
    {
        using namespace rtf::com::utils;

        // Initialize Method
        /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
        RTF_DEBUG_LOG(logger_, "[RTFCOM] Registering method '", uri_, "'...");
        if (!InitializeMethod<vrtf::vcc::api::VccMethodReturnType<Response>>()) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
            RTF_ERROR_LOG(logger_, "[RTFCOM] Failed to register method '", uri_, "'");
            return false;
        }
        std::function<vrtf::vcc::api::VccMethodReturnType<Response>(rtf::com::e2e::Result, Request)> methodHandler =
            [callback] (rtf::com::e2e::Result result,
                        Request request) -> vrtf::vcc::api::VccMethodReturnType<Response> {
                Response response;
                MethodServerResult serverResult;
                serverResult.SetE2EResult(result);
                callback(request, response, serverResult);
                vrtf::core::Promise<Response> promise;
                promise.set_value(response);
                return vrtf::vcc::api::VccMethodReturnType<Response>(std::move((promise.get_future().GetResult())),
                                                                               serverResult.IsUsingIncorrectE2EId());
            };
        std::unordered_map<AdapterProtocol, std::shared_ptr<Entity>> entityMapBack;
        {
            std::lock_guard<std::mutex> lock{entityMapMutex_};
            entityMapBack = entityMap_;
        }
        for (auto& entityMapIterator : entityMapBack) {
            const auto& protocol = entityMapIterator.first;
            const auto& entity   = entityMapIterator.second;
            const auto& skeleton = entity->skeleton;
            // Register method
            VccThreadPoolPair tPair {VccThreadPoolType::RTF_COM, threadPool_};
            skeleton->RegisterMethod(entityId_, methodHandler, tPair);
            // Offer entity
            if (!isUsingDefaultConfig_) {
                OfferEntity(entity, protocol);
            } else {
                if (vrtf::vcc::utils::stats::DataStatistician::GetInstance()->IsEnable()) {
                    skeleton->EnableDataStatistics(vrtf::vcc::utils::stats::DataStatistician::GetInstance()->Period());
                }
            }
        }
        /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
        RTF_DEBUG_LOG(logger_, "[RTFCOM] Method '", uri_, "' is registered");
        return true;
    }

    /**
     * @brief Publish a message
     * @param[in] message    Message data
     * @return void
     */
    template <typename EventDataType>
    rtf::com::ErrorCode Publish(const EventDataType& message, internal::SampleInfoImpl& info)
    {
        using namespace rtf::com::utils;
        if (!IsValid()) {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            RTF_ERROR_LOG_SPR(logger_, "RosSkeletonAdapter_Publish", vrtf::vcc::api::types::LOG_LIMIT_SECOND_60,
                "[RTFCOM] Cannot publish message to '", uri_, "', service is not avaliable");
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return rtf::com::ErrorCode::ERROR;
        }
        std::unordered_map<AdapterProtocol, std::shared_ptr<Entity>> entityMapBack;
        {
            std::lock_guard<std::mutex> lock{entityMapMutex_};
            entityMapBack = entityMap_;
        }
        rtf::com::ErrorCode errorCode = rtf::com::ErrorCode::OK;
        for (auto& entityMapIterator : entityMapBack) {
            const auto& entity          = entityMapIterator.second;
            const auto& isEntityOffered = entity->isOffered;
            if (isUsingDefaultConfig_ || isEntityOffered) {
                const auto& skeleton = entity->skeleton;
                if (!skeleton->Send<EventDataType>(message, entityId_, info).HasValue()) {
                    errorCode = rtf::com::ErrorCode::ERROR;
                }
            }
        }
        return errorCode;
    }

    /**
     * @brief Stop responding the ros uri that adapter handles
     * @return void
     */
    void Shutdown(bool isEntityLevel = true) noexcept;

    /**
     * @brief Allocate Buffer for using RawMemory
     *
     * @param[in] size    The size of buffer will be allocated
     * @return RawMemory  The buffer was allocated
     */
    RawMemory AllocateRawMemory(std::size_t size) noexcept;

    /**
     * @brief  Free the allocated buffer last time
     *
     * @param[inout] buffer  the buffer will be free
     */
    void DeallocateRawMemory(RawMemory& buffer) noexcept;

    /**
     * @brief Publish a raw buffer
     *
     * @param[inout] buffer The buffer will be sent
     */
    rtf::com::ErrorCode PubRawMemory(RawMemory& buffer) noexcept;

    rtf::com::AppState GetSomeipAppInitState() const noexcept;
    void RegisterEraseSkeletonFunc(rtf::com::internal::EraseEntityHandler const &func);
    ReturnCode WaitForFlush(const std::uint32_t waitMs) noexcept;
private:
    struct Entity {
        std::shared_ptr<VccSkeleton>  skeleton;
        std::shared_ptr<EntityConfig> config;
        bool isOffered;
    };
    std::string     uri_;
    EntityId        entityId_;
    AdapterType     type_;
    MaintainConfig  maintainConfig_;
    std::shared_ptr<VccThreadPool> threadPool_;
    std::unordered_map<AdapterProtocol, std::shared_ptr<Entity>> entityMap_;
    std::mutex entityMapMutex_;

    bool isUsingDefaultConfig_;
    bool isInitialized_;
    bool isOffered_;
    bool isValid_;
    std::atomic<bool> isShutDown_ {false};
    // maintain SomeipJsonHelper/ProlocManager life cycle
    std::shared_ptr<rtf::com::utils::SomeipJsonHelper> someipJsonHelperPtr_;
    std::shared_ptr<rtf::com::utils::ProlocManager> prolocManager_;
    vrtf::driver::proloc::ProlocEntityIndex prolocIndex_;
    std::shared_ptr<ara::godel::common::log::Log> logger_;
    rtf::com::AppState someipAppInitState_ {rtf::com::AppState::APP_DEREGISTERED};
    rtf::com::internal::EraseEntityHandler eraseSkeletonFunc_;
    std::mutex eraseSkeletonMutex_;
    std::shared_ptr<vrtf::driver::someip::SomeipInterface> interface_;
    PoolHandle poolHandle_ = nullptr;
    std::shared_ptr<vrtf::vcc::utils::DpAdapterHandler> dpHandler_ {nullptr};
    bool enableDirectProcess_ = false;
    bool plogSendEnableFlag_ = false;
    std::shared_ptr<vrtf::vcc::EventSkeleton> ddsEventSkeleton_;

    /**
     * @brief Parse config at the begining
     * @param[in] role  The attr of the entity
     */
    bool ParseConfig(const EntityAttr& attr) noexcept;

    void SyncorizeEntityConfig(void) noexcept;

    /**
     * @brief Initialize entity for current config
     * @param[in] attr  The entity attributes
     *
     * @return Entity initialization result
     */
    bool InitializeEntity(const EntityAttr& attr) noexcept;

    /**
     * @brief Initialize event
     *
     * @return Event initialization result
     */
    bool InitializeEvent(void) noexcept;

    template <class VccResponse>
    bool InitializeMethod(void) noexcept
    {
        using namespace rtf::com::utils;

        bool result = false;
        std::unordered_map<AdapterProtocol, std::shared_ptr<Entity>> entityMapBack;
        {
            std::lock_guard<std::mutex> lock{entityMapMutex_};
            entityMapBack = entityMap_;
        }
        for (auto& entityMapIterator : entityMapBack) {
            const auto& protocol = entityMapIterator.first;
            const auto& entity   = entityMapIterator.second;

            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
            RTF_DEBUG_LOG(logger_, "[RTFCOM] Initializing method '", uri_, "'...");
            const auto& methodInfo = std::static_pointer_cast<VccMethodInfo>((entity->config)->entityInfo);
            if (methodInfo == nullptr) {
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
                RTF_ERROR_LOG(logger_, "[RTFCOM] Cannot find method configuration for '", uri_, "'");
                result = false;
                break;
            }

            const auto& skeleton   = entity->skeleton;
            const auto& driverType = protocolDriverMap_.at(protocol);

            // Additional operation is needed for SOME/IP protocol
            if ((driverType == VccDriverType::SOMEIPTYPE) && (!AddServiceToSOMEIPD())) {
                result = false;
                break;
            }
            if (!(skeleton->InitializeMethod<VccResponse>({{ driverType, methodInfo }}))) {
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
                RTF_ERROR_LOG(logger_, "[RTFCOM] Failed to initialize method '", uri_, "'");
                result = false;
                break;
            }
            if (entity->config->trafficCrtlPolicy != nullptr) {
                if (skeleton->SetMethodTrafficCtrl(entity->config->trafficCrtlPolicy, methodInfo->GetEntityId())) {
                    /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
                    RTF_INFO_LOG(logger_, "[RTFCOM] method ", uri_, " enable traffic control successful");
                } else {
                    /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
                    RTF_ERROR_LOG(logger_, "[RTFCOM] method ", uri_, " enable traffic control failed");
                }
            }
            result = true;
        }
        return result;
    }
    /**
     * @brief Offer entity
     * @param[in] protocol The entity of the service
     * @param[in] protocol The protocol that service offered on
     */
    void OfferEntity(const std::shared_ptr<Entity>& entity,
                     const AdapterProtocol& protocol) noexcept;

    /**
     * @brief Stop offer entity
     */
    void StopOfferEntity(void) noexcept;

    /* Static map <key = serviceName, value = entityList{ uri, ... }>
    Since there would be multiple skeletons providing same service across RosSkeletonAdapters,
    we need this map to count literal how many VccSkeleton are providing the service.
    This mechanism ensures the service would be stopped on the right time:
    1. When a VccSkeleton offers a service, it should add its uri to the entity list.
    2. On the contrary, when a VccSkeleton shutsdown, it should NOT simply stop offering the
    service, instead, it should check and erase its uri from the entity list.
    3. When the entity list is empty, the service should be stopped. */
    void AddEntityToServiceMap(const std::string& serviceName) noexcept;
    void RemoveEntityFromServiceMap(const std::string& serviceName) noexcept;
    static std::mutex serviceMapMutex_;
    static std::unordered_map<std::string, std::unordered_set<std::string>> serviceMap_;
    const std::unordered_map<AdapterProtocol, VccDriverType> protocolDriverMap_ = {
        { AdapterProtocol::UNKNOWN, VccDriverType::INVALIDTYPE },
        { AdapterProtocol::DDS,     VccDriverType::DDSTYPE     },
        { AdapterProtocol::SOMEIP,  VccDriverType::SOMEIPTYPE  },
        { AdapterProtocol::PROLOC,      VccDriverType::PROLOCTYPE},
    };
    /**
     * @brief Add an service info to someipd
     *
     * @return true  successfully add to someipd
     * @return false falied to add service info into someipd
     */
    bool AddServiceToSOMEIPD(void) noexcept;

    /**
     * @brief Delete service info from someipd if the last user of the service is called by shutdown
     */
    void DeleteEventServiceFromSOMEIPD(void) noexcept;
    void ParseProlocConfig(
        const std::shared_ptr<EntityConfig> &ddsConfig, const std::shared_ptr<EntityConfig> &someipConfig) noexcept;
    void AddProlocInfo(const std::shared_ptr<EntityConfig> &prolocInfo) noexcept;
    void EraseRosSkeletonFromHolder();
    bool InitDirectProcess(VccDriverType driverType, const std::shared_ptr<VccEventInfo> &eventInfo) noexcept;
    RawMemory GetRawMemoryWithDirectProcess(std::size_t size);
};
} // namspace adapter
} // namspace com
} // namspace rtf
#endif // RTF_COM_ADAPTER_ROS_SKELETON_ADAPTER_H

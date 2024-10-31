/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: EventHandler in CM
 * Create: 2019-11-19
 */
#ifndef VRTF_VCC_DRIVER_EVENTHANDLER_H
#define VRTF_VCC_DRIVER_EVENTHANDLER_H
#include <vector>
#include <memory>
#include <unordered_map>
#include "vrtf/vcc/event_proxy_call.h"
#include "vrtf/driver/dds/mbuf.h"
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/utils/plog_info.h"
#include "vrtf/vcc/utils/latency_analysis.h"
#include "vrtf/vcc/utils/tlv_helper.h"
#include "vrtf/vcc/utils/stats/stats.h"
#include "vrtf/driver/proloc/proloc_driver_types.h"
#include "vrtf/driver/proloc/proloc_memory_manager.h"
#include "vrtf/vcc/api/subscriber_listener.h"
namespace vrtf {
namespace vcc {
class EventProxy;
namespace driver {
class EventCache {
public:
    EventCache(uint8_t* data, const size_t& size, Mbuf *mbufPtr, const vcc::api::types::SampleTimeInfo& takeInfo)
        : value_(data),
          size_(size),
          mbufPtr_(mbufPtr),
          takeInfo_(takeInfo) {}
    ~EventCache(void) = default;
    const uint8_t* GetData() const
    {
        return value_;
    }
    const Mbuf *GetMbufPtr() const
    {
        return mbufPtr_;
    }
    size_t GetSize() const
    {
        return size_;
    }

    /**
     * @brief Get sample take info
     * @details Get sample take info
     *
     * @return size_t msg
     */
    const vrtf::vcc::api::types::SampleTimeInfo& GetSampleTimeInfo() const
    {
        return takeInfo_;
    }

    /**
     * @brief Set SecOC deauth result
     * @param SecOC deauth res
     */
    void SetIdentityResult(bool res)
    {
        identityRes_ = res;
    }

    /**
     * @brief Get SecOC deauth result of this event cache
     * @return deauth result of this event cache(default true)
     */
    bool GetIdentityResult(void) const
    {
        return identityRes_;
    }

    uint8_t *value_;
private:
    size_t size_;
    Mbuf *mbufPtr_;
    vrtf::vcc::api::types::SampleTimeInfo takeInfo_;
    bool identityRes_ {true};
};
using EventCacheContainer = std::vector<EventCache>;
using E2EResultContainer = std::vector<vrtf::com::e2exf::Result>;
using CacheStatus = vrtf::vcc::api::types::CacheStatus;
using StatisticInfo = vrtf::vcc::api::types::StatisticInfo;
class EventHandler : public std::enable_shared_from_this<EventHandler> {
public:
    EventHandler() = default;
    virtual ~EventHandler() = default;
    virtual bool SendEvent(uint8_t * const data, const size_t length,
                           vrtf::vcc::api::types::internal::SampleInfoImpl& info) = 0;
    virtual void ReadEvent(EventCacheContainer &data, E2EResultContainer& e2eStatus, std::int32_t size) = 0;
    virtual uint8_t* AllocateBuffer(const uint32_t length) = 0;
    virtual void ReturnLoan(const uint8_t *data) = 0;
    virtual void ServerReturnLoan(const uint8_t *data) = 0;
    virtual void SetReceiveHandler(vrtf::vcc::api::types::EventHandleReceiveHandler handler) = 0;
    virtual void SetSubscriberListener(const vrtf::vcc::api::types::internal::ListenerRegisterParams& handler) = 0;
    virtual vrtf::serialize::SerializeType GetSerializeType() = 0;
    virtual std::string GetDriverName(void) = 0;
    virtual bool EnableEvent() = 0;
    virtual void UnsubscribeEvent() = 0;
    virtual std::size_t GetE2EHeaderSize() = 0;
    virtual bool IsEnableDp() = 0;
    virtual bool HasMessageTake() noexcept = 0;
    virtual CacheStatus GetEventCacheStatus(void) = 0;
    virtual void GetProtocolStatInfo(StatisticInfo& statInfo) = 0;
    // Dp raw data for mbuf and queue event.
    virtual bool IsDpRawData() = 0;
    virtual std::string QueryDataStatistics() = 0;
    virtual size_t GetEventUid() const = 0;
    // only use in proloc
    virtual void SetProlocMemoryManager(std::shared_ptr<vrtf::driver::proloc::ProlocMemoryManager> ptr) = 0;
    virtual void SetEventProxyStatus(std::shared_ptr<vrtf::vcc::utils::stats::EventProxyStats> stats) noexcept = 0;
    virtual void SetEventSkeletonStatus(std::shared_ptr<vrtf::vcc::utils::stats::EventSkeletonStats> stats) noexcept = 0;
    // only use in dds
    virtual vrtf::vcc::api::types::ReturnCode WaitForFlush(const std::uint32_t waitMs) noexcept = 0;
    // for callback
    virtual void SetEventProxy(vrtf::vcc::EventProxyCall *proxy) noexcept = 0;

};
}
}
}

#endif

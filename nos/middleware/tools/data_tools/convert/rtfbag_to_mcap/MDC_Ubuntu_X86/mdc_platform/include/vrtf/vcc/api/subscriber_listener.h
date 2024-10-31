/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This provide the interface of SubscriberListener.
 * Create: 2022-06-18
 */

#ifndef VRTF_VCC_SUBSCRIBER_LISTENER_H
#define VRTF_VCC_SUBSCRIBER_LISTENER_H
#include <bitset>
#include "vrtf/vcc/api/types.h"
#include "ara/core/variant.h"
namespace vrtf {
namespace vcc {
namespace api {
namespace types {
using ListenerMaskType = std::bitset<32>;
class ListenerMask final : public ListenerMaskType {
public:
    using ListenerMaskType::ListenerMaskType;
    static ListenerMask All() noexcept { return ListenerMask{0x3U}; };
    static ListenerMask SampleLost() noexcept { return ListenerMask{1U}; }
    static ListenerMask SampleTimeOut() noexcept { return ListenerMask{2U}; }
    ~ListenerMask() = default;
};
class ListenerTimeoutInfo {
public:
    void SetMsgId(const std::uint64_t msgId) noexcept { msgId_ = msgId; }
    std::uint64_t GetMsgId() const noexcept { return msgId_; }
    void SetPubPid(const std::uint32_t pubPid) noexcept { pubUid_ = pubPid; }
    std::uint32_t GetPubPid() const noexcept { return pubUid_; }
    void SetSubPid(const std::uint32_t subPid) noexcept { subUid_ = subPid; }
    std::uint32_t GetSubPid() const noexcept { return subUid_; }
    void SetSampleIndex(const std::string& sampleIndex) noexcept { sampleIndex_ = sampleIndex; }
    std::string GetSampleIndex() const noexcept { return sampleIndex_; }
private:
    std::uint64_t msgId_ {0};
    std::string ipAddress_;
    std::uint32_t pubUid_ {0};
    std::uint32_t subUid_ {0};
    std::string sampleIndex_;
};
class SubscriberListener;
namespace internal {
class VccListenerParams final {
public:
    void SetListenerPointer(
        const std::shared_ptr<vrtf::vcc::api::types::SubscriberListener>& ptr) noexcept { sampleListener_ = ptr; }
    std::shared_ptr<vrtf::vcc::api::types::SubscriberListener> GetListenerPointer() const noexcept
    {
        return sampleListener_;
    }
    void SetListenerMask(const vrtf::vcc::api::types::ListenerMask& mask) noexcept { listenerMask_ = mask; }
    vrtf::vcc::api::types::ListenerMask GetListenerMask() const noexcept { return listenerMask_; }
private:
std::shared_ptr<vrtf::vcc::api::types::SubscriberListener> sampleListener_;
vrtf::vcc::api::types::ListenerMask listenerMask_;
};
struct SampleLostInfo {
    std::uint64_t totalCount {0};
    std::uint64_t totalChangeCount {0};
};
enum class ListenerSource : uint8_t {
    USER_TRIGGER = 0U,
    RTF_TOOLS_TRIGGER = 1U
};
using ListenerCallbackType = ara::core::Variant<SampleLostInfo, ListenerTimeoutInfo>;
using ListenerCallbackFunction =
    std::function<void(const ListenerCallbackType&, const vrtf::vcc::api::types::DriverType)>;
class ListenerRegisterParams final {
public:
    void SetListenerCallbackFunction(const ListenerCallbackFunction& handler) noexcept { handler_ = handler; }
    ListenerCallbackFunction GetListenerCallbackFunction() const noexcept { return handler_; }
    void SetTimeoutThreshold(const std::uint32_t millisecond) noexcept { millisecond_ = millisecond; }
    std::uint32_t GetTimeoutThreshold() const noexcept { return millisecond_; }
    void SetListenerMask(const vrtf::vcc::api::types::ListenerMask& mask) noexcept { listenerMask_ = mask; }
    vrtf::vcc::api::types::ListenerMask GetListenerMask() const noexcept { return listenerMask_; }
    bool operator==(std::nullptr_t) const noexcept;
    bool operator!=(std::nullptr_t) const noexcept;
private:
    ListenerCallbackFunction handler_;
    std::uint32_t millisecond_ = 0U;
    vrtf::vcc::api::types::ListenerMask listenerMask_ {vrtf::vcc::api::types::ListenerMask::All()};
};
}
class SubscriberListener {
public:
    SubscriberListener();
    virtual ~SubscriberListener() = default;
    virtual void OnSampleLost(
        std::uint64_t totalCount, std::uint64_t totalChangeCount, vrtf::vcc::api::types::DriverType type);
    virtual void OnSampleTimeout(const vrtf::vcc::api::types::ListenerTimeoutInfo& sampleInfo,
        const vrtf::vcc::api::types::DriverType type) noexcept;
    void SetTimeoutThreshold(const std::uint32_t time) noexcept { timeoutThreshold_ = time; }
    std::uint32_t GetTimeoutThreshold() const noexcept { return timeoutThreshold_; }
private:
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    std::uint32_t timeoutThreshold_ {0U};
};
}
}
}
}

#endif //VRTF_VCC_SUBSCRIBER_LISTENER_H

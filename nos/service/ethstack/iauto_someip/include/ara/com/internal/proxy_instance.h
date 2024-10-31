#ifndef ARA_COM_INTERNAL_PROXY_INSTANCE_H_
#define ARA_COM_INTERNAL_PROXY_INSTANCE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <deque>
#include <vector>
#include <mutex>
#include <atomic>
#include "ara/core/promise.h"
#include "ara/core/instance_specifier.h"
#include "ara/com/types.h"
#include "ara/com/com_error_domain.h"
#include "ara/com/instance_identifier.h"
#include "ara/com/internal/internal_types.h"
#include "ara/com/internal/instance_base.h"
#include "ara/com/serializer/error_code_transformation.h"
#include "ara/com/serializer/transformation_props.h"


namespace ara {
namespace core {
namespace extend {
class LooperQueue;
}
}
}

namespace ara {
namespace com {
namespace runtime {

class TransportBindingSession;
class TransportRoutingProxy;

template <typename Output>
struct PromiseTag : public TagBase {
    ara::core::Promise<Output> promise_;
};

class ProxyInstance : public InstanceBase {
public:
    ProxyInstance(const std::string& service_info);
    virtual ~ProxyInstance();

    static ara::com::ServiceHandleContainer<ara::com::HandleType> FindService(const std::string& service_info, ara::com::InstanceIdentifier instance);

    static ara::com::ServiceHandleContainer<ara::com::HandleType> FindService(const std::string& service_info, ara::core::InstanceSpecifier instance);

    static ara::com::FindServiceHandle StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, const std::string& service_info,
                                                        ara::com::InstanceIdentifier instance_id);

    static ara::com::FindServiceHandle StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, const std::string& service_info,
                                                        ara::core::InstanceSpecifier instance);

    static void StopFindService(ara::com::FindServiceHandle handle, const std::string& service_info);

    // TODO
    // void updateHandleType(const ara::com::HandleType& handle_type, event_queue*, method_queue*);
    // void updateHandleType(const ara::com::HandleType& handle_type, context*);

    void updateHandleType(const ara::com::HandleType& handle_type);
    ara::com::HandleType getHandle();

    int32_t methodSerializationType(uint32_t idx);
    int32_t eventSerializationType(uint32_t idx);
    int32_t fieldSerializationType(uint32_t idx);


    /* =============== common =================================================================== */
    // method(field-setter/field-getter)
    // event(field-notifier)
    using ProxyResponseCallback = std::function<void(int32_t return_code, const std::shared_ptr<BufferList>& payload,
                                                     std::shared_ptr<TagBase>& tag)>;

    using F = std::function<void(const std::shared_ptr<BufferList>&)>;
    /* =============== common =================================================================== */

    /* =============== method =================================================================== */
    void setMethodResponseCallback(uint32_t idx, const ProxyResponseCallback& callback);

    int32_t sendMethodAsyncRequest(uint32_t method_idx, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);
    /* =============== method =================================================================== */

    /* =============== event =================================================================== */
    void setEventSubscriptionStateChangeHandler(uint32_t event_idx, ara::com::SubscriptionStateChangeHandler handler);
    void unsetEventSubscriptionStateChangeHandler(uint32_t event_idx);
    void setEventReceiveHandler(uint32_t event_idx, ara::com::EventReceiveHandler handler);
    void unsetEventReceiveHandler(uint32_t event_idx);
    int32_t subscribeEvent(uint32_t event_idx, std::size_t maxSampleCount);
    int32_t unsubscribeEvent(uint32_t event_idx);
    ara::com::SubscriptionState getEventSubscriptionState(uint32_t event_idx);
    size_t getEventFreeSampleCount(uint32_t event_idx);
    size_t getEventNewSamples(uint32_t event_idx, size_t maxNumberOfSamples, std::vector<std::shared_ptr<BufferList>>& buffer_lists);
    /* =============== event =================================================================== */

    /* =============== field =================================================================== */
    void setFieldSetResponseCallback(uint32_t idx, const ProxyResponseCallback& callback);

    void setFieldGetResponseCallback(uint32_t idx, const ProxyResponseCallback& callback);

    int32_t sendFieldSetAsyncRequest(uint32_t field_idx, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);

    int32_t sendFieldGetAsyncRequest(uint32_t field_idx, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);

    void setFieldSubscriptionStateChangeHandler(uint32_t field_idx, ara::com::SubscriptionStateChangeHandler handler);
    void unsetFieldSubscriptionStateChangeHandler(uint32_t field_idx);
    void setFieldReceiveHandler(uint32_t field_idx, ara::com::EventReceiveHandler handler);
    void unsetFieldReceiveHandler(uint32_t field_idx);
    int32_t subscribeField(uint32_t field_idx, std::size_t maxSampleCount);
    int32_t unsubscribeField(uint32_t field_idx);
    ara::com::SubscriptionState getFieldSubscriptionState(uint32_t field_idx);

    size_t getFieldFreeSampleCount(uint32_t field_idx);
    size_t getFieldNewSamples(uint32_t field_idx, size_t maxNumberOfSamples, std::vector<std::shared_ptr<BufferList>>& buffer_lists);
    /* =============== field =================================================================== */

private:
    struct SessionKey {
        SessionKey(const std::shared_ptr<TransportBindingSession>& session);
        bool operator==(const SessionKey& other) const;
        bool operator<(const SessionKey& other) const;
        std::shared_ptr<TransportBindingSession> session_;
    };

    struct MethodData {
        MethodData();
        ProxyResponseCallback callback;
        std::mutex mutex;
        std::map<SessionKey, std::shared_ptr<TagBase>> session_tag_map;

        void* rep_stat_handle;
        void* req_stat_handle;
    };

    struct EventData {
        EventData(const std::string& name = "");
        ara::com::SubscriptionStateChangeHandler subscription_state_changed_handler;
        ara::com::EventReceiveHandler event_receive_handler;
        bool event_receive_handler_set;
        std::condition_variable cond_var;
        std::mutex mut;
        std::atomic<uint32_t> subscription_state;
        std::mutex mutex;
        std::size_t max_sample_count;
        std::deque<std::shared_ptr<BufferList>> event_caches;
        std::shared_ptr<ara::core::extend::LooperQueue> event_queue_;    // TODO set timer to clear queue
        std::string name_;

        void* stat_handle;
    };

    struct FieldData {
        MethodData setter;
        MethodData getter;
        EventData notifier;
    };

private:
    std::shared_ptr<TransportRoutingProxy> routing_proxy_;
    ara::com::HandleType  handle_type_;
    MethodData* method_data_list_;
    std::vector<std::unique_ptr<EventData>> event_data_list_;
    FieldData* field_data_list_;
    std::shared_ptr<ara::core::extend::LooperQueue> response_queue_; // TODO set timer to clear queue
    static std::shared_ptr<ara::core::extend::LooperQueue> findservice_handle_queue_; // TODO set timer to clear queue
    static struct StaticInitializer {
        StaticInitializer();
    } static_member_initializer_;

};

}  // namespace runtime
}  // namespace com
}  // namespace ara

#endif // ARA_COM_INTERNAL_PROXY_INSTANCE_H_
/* EOF */

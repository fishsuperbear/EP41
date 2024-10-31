#ifndef ARA_COM_INTERNAL_SKELETON_INSTANCE_H_
#define ARA_COM_INTERNAL_SKELETON_INSTANCE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <vector>
#include <map>
#include <memory>
#include "ara/core/promise.h"
#include "ara/core/instance_specifier.h"
#include "ara/com/types.h"
#include "ara/com/internal/internal_types.h"
#include "ara/com/internal/instance_base.h"
#include "ara/com/serializer/error_code_transformation.h"
#include "ara/com/serializer/transformation_props.h"

namespace ara {
namespace core {
namespace extend {
class LooperQueue;
class MainlooperContext;
}
}
}


namespace ara {
namespace com {
namespace runtime {

class TransportBindingSession;
struct TransportBindingClientHandle;
struct TransportServiceInterface;
class TransportRoutingSkeleton;

struct OneWayTag : public TagBase {
};

class SkeletonInstance : public InstanceBase {
public:
    SkeletonInstance(const std::string& service_info, ara::com::InstanceIdentifier instanceID, ara::com::MethodCallProcessingMode mode);
    SkeletonInstance(const std::string& service_info, ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode);

    // TODO
    // SkeletonInstance(const std::string& service_info, ara::com::InstanceIdentifier instanceID, request_queue_*);
    // SkeletonInstance(const std::string& service_info, ara::core::InstanceSpecifier instance_specifier, request_queue_*);


    virtual ~SkeletonInstance();

    void start();

    void offerService();

    void stopOfferService();

    bool processNextMethodCall();

    int32_t eventSerializationType(uint32_t idx);
    int32_t fieldSerializationType(uint32_t idx);

    using SkeletonRequestAsyncCallback = std::function<void(const std::shared_ptr<BufferList>& payload,
                                                            std::shared_ptr<TagBase>& tag, int32_t e2e_result)>;
    void setMethodAsyncRequestCallback(uint32_t idx, const SkeletonRequestAsyncCallback& callback);

    void setFieldSetAsyncRequestCallback(uint32_t idx, const SkeletonRequestAsyncCallback& callback);

    void setFieldGetAsyncRequestCallback(uint32_t idx, const SkeletonRequestAsyncCallback& callback);


    int32_t sendMethodResponse(uint32_t method_idx, int32_t error_code, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);
    int32_t sendFieldSetResponse(uint32_t field_idx, int32_t error_code, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);
    int32_t sendFieldGetResponse(uint32_t field_idx, int32_t error_code, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);

    int32_t sendEventNotify(uint32_t event_idx, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);

    int32_t sendFieldNotify(uint32_t field_idx, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag);

private:
    struct SessionTag : public TagBase {
        SessionTag(const std::shared_ptr<TransportBindingClientHandle>& client_handle,
            const std::shared_ptr<TransportBindingSession>& binding_session, uint32_t skeleton_idx);
        std::shared_ptr<TransportBindingClientHandle> client_handle_;
        std::shared_ptr<TransportBindingSession> binding_session_;
        uint32_t skeleton_idx_;
    };

    struct MethodData {
        SkeletonRequestAsyncCallback callback;
        void* rep_stat_handle;
        void* req_stat_handle;
    };

    struct EventData {
        void* notifier_handle;
    };

    struct FieldData {
        MethodData setter;
        MethodData getter;
        void* notifier_state_handle;
        // EventData notifier;
    };

private:
    ara::com::MethodCallProcessingMode mode_;
    std::shared_ptr<TransportRoutingSkeleton> routing_skeleton_;
    MethodData* method_data_list_;
    FieldData* field_data_list_;
    EventData* event_data_list_;
    std::shared_ptr<ara::core::extend::MainlooperContext> main_looper_context_;
    std::shared_ptr<ara::core::extend::LooperQueue> request_queue_;
    std::atomic<uint32_t> skeleton_state_;
    std::condition_variable cond_var_;
    std::mutex mut_;
};

}  // namespace runtime
}  // namespace com
}  // namespace ara

#endif // ARA_COM_INTERNAL_SKELETON_INSTANCE_H_
/* EOF */

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: This file provides an interface related to communication management.
 * Create: 2020-08-03
 */
#ifndef ARA_COM_SKELETON_METHOD_ADAPTER_H
#define ARA_COM_SKELETON_METHOD_ADAPTER_H
#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "vrtf/vcc/internal/traffic_crtl_policy.h"
namespace ara {
namespace com {
namespace internal {
namespace skeleton {
namespace method {
namespace impl {
class MethodAdapterImpl {
public:
    MethodAdapterImpl(const std::shared_ptr<vrtf::vcc::Skeleton>& skeleton, EntityId entityId)
        : skeleton_(skeleton), entityId_(entityId)
    {
    }

    virtual ~MethodAdapterImpl(void)
    {
    }

    // Internal interface!!! Prohibit to use by Application!!!!
    EntityId GetEntityId(void) const
    {
        return entityId_;
    }

    bool SetThreadGroup(const std::shared_ptr<ara::com::ThreadGroup>& threadGroup)
    {
        return skeleton_->SetMethodThreadGroup(threadGroup, entityId_);
    }
protected:
    std::shared_ptr<vrtf::vcc::Skeleton> skeleton_;
    EntityId entityId_ = UNDEFINED_ENTITYID;

    bool SetTrafficCtrl(const std::shared_ptr<rtf::TrafficCtrlPolicy>& policy)
    {
        return skeleton_->SetMethodTrafficCtrl(policy, GetEntityId());
    }
};
}

class MethodAdapter : public impl::MethodAdapterImpl {
public:
    MethodAdapter(const std::shared_ptr<vrtf::vcc::Skeleton>& skeleton, EntityId entityId)
        : MethodAdapterImpl(skeleton, entityId)
    {
    }
    ~MethodAdapter() = default;

    bool SetTrafficCtrl(const std::shared_ptr<rtf::TrafficCtrlPolicy>& policy)
    {
        return MethodAdapterImpl::SetTrafficCtrl(policy);
    }

    bool SetThreadGroup(const std::shared_ptr<ara::com::ThreadGroup>& threadGroup)
    {
        return MethodAdapterImpl::SetThreadGroup(threadGroup);
    }
};
}
}
}
}
}

#endif

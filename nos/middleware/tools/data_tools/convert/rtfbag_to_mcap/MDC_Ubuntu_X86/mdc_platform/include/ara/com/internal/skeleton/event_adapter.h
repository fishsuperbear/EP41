/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: This file provides an interface related to communication management.
 * Create: 2019-07-01
 */
#ifndef ARA_COM_SKELETON_EVENT_ADAPTER_H
#define ARA_COM_SKELETON_EVENT_ADAPTER_H
#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "vrtf/vcc/internal/traffic_crtl_policy.h"
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
#ifndef COVERITY_TAINT_SET_DEFINITION
#define COVERITY_TAINT_SET_DEFINITION
/**
 * @brief Function for Stain Modeling
 * @details The function is used only when the compilation macro AOS_TAINT is enabled.
 */
static void Coverity_Tainted_Set(void *buf){}
#endif
#endif
namespace ara {
namespace com {
namespace internal {
namespace skeleton {
namespace event {
namespace impl {
class EventAdapterImpl {
public:
    EventAdapterImpl(const std::shared_ptr<vrtf::vcc::Skeleton> &skeleton, EntityId entityId)
        : skeleton_(skeleton), entityId_(entityId)
    {
    }

    virtual ~EventAdapterImpl(void)
    {
    }

    // Internal interface!!! Prohibit to use by Application!!!!
    EntityId GetEntityId(void) const
    {
        return entityId_;
    }

protected:
    std::shared_ptr<vrtf::vcc::Skeleton> skeleton_;
    EntityId entityId_ = UNDEFINED_ENTITYID;

    template<class SampleType>
    ara::com::SampleAllocateePtr<SampleType> Allocate(void)
    {
        return skeleton_->Allocate<SampleType>();
    }

    template<class SampleType>
    typename std::enable_if<IsRawMemory<SampleType>::value, RawMemory>::type Allocate(const size_t &size)
    {
        return skeleton_->AllocateRawBuffer(size, entityId_);
    }

    template<class SampleType>
    typename std::enable_if<IsRawMemory<SampleType>::value>::type Send(const SampleType &&data)
    {
        RawMemory rawBuffer = data;
        SampleInfoImpl sampleInfo = CreateSampleInfo();
        static_cast<void>(skeleton_->PubRawBuffer(rawBuffer, entityId_, sampleInfo));
    }

    template<class SampleType>
    typename std::enable_if<!IsRawMemory<SampleType>::value>::type Send(const SampleType &data)
    {
        SampleInfoImpl sampleInfo = CreateSampleInfo();
        static_cast<void>(skeleton_->Send<SampleType>(data, entityId_, sampleInfo));
    }

    bool SetTrafficCtrl(const std::shared_ptr<rtf::TrafficCtrlPolicy> &policy)
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void*)&policy);
#endif
        return skeleton_->SetEventTrafficCtrl(policy, entityId_);
    }
private:
    SampleInfoImpl CreateSampleInfo() const
    {
        SampleInfoImpl sampleInfo;
        if (PlogInfo::GetSendFlag()) {
            sampleInfo.plogInfo_ = PlogInfo::CreatePlogInfo(vrtf::vcc::utils::CM_SEND);
            if (sampleInfo.plogInfo_ != nullptr) {
                sampleInfo.plogInfo_->WriteTimeStamp(PlogServerTimeStampNode::USER_SEND_EVENT, PlogDriverType::COMMON);
            }
        }
        return sampleInfo;
    }
};
}

template<class SampleType>
class EventAdapter : public impl::EventAdapterImpl {
public:
    EventAdapter(const std::shared_ptr<vrtf::vcc::Skeleton> &skeleton, EntityId entityId)
        : EventAdapterImpl(skeleton, entityId)
    {
    }
    ~EventAdapter() = default;
    /**
     * @brief allocate event data type ptr
     * @details allocate event data type ptr
     *
     * @return Return sample allocate ptr of event type
     */
    ara::com::SampleAllocateePtr<SampleType> Allocate(void)
    {
        return EventAdapterImpl::Allocate<SampleType>();
    }

    /**
     * @brief allocate shm memory for rawmemory type
     * @details allocate shm memory for rawmemory type
     *
     * @param size size of shm memory
     * @return Return raw memory data type
     */
    SampleType Allocate(const size_t size)
    {
        return EventAdapterImpl::Allocate<SampleType>(size);
    }

    /**
     * @brief Send data created by user
     * @details Send data created by user
     *
     * @param data dataType except rawBuffer
     */
    void Send(const SampleType &data)
    {
        EventAdapterImpl::Send<SampleType>(data);
    }

    /**
     * @brief Send rawmemory data type
     * @details Send rawmemory data type
     *
     * @param data raw buffer
     */
    void Send(const SampleType &&data)
    {
        EventAdapterImpl::Send<SampleType>(std::move(data));
    }

    void Send(ara::com::SampleAllocateePtr<SampleType> data)
    {
        if (data == nullptr) {
            return;
        }
        std::shared_ptr<typename std::decay<SampleType>::type> sendptr(data.release());
        EventAdapterImpl::Send<SampleType>(*sendptr);
    }

    bool SetTrafficCtrl(const std::shared_ptr<rtf::TrafficCtrlPolicy> &policy)
    {
        return EventAdapterImpl::SetTrafficCtrl(policy);
    }
};
}
}
}
}
}

#endif

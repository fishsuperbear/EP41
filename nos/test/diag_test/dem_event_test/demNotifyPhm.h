#pragma once

#include "cm/include/skeleton.h"
#include "cm/include/proxy.h"
#include "idl/generated/phmPubSubTypes.h"
#include "idl/generated/diagPubSubTypes.h"
#include <memory>
#include <functional>

class DiagServerEventPub
{
public:
    DiagServerEventPub();
    virtual ~DiagServerEventPub();

    void init();
    void deInit();
    void sendFaultEvent(uint32_t faultKey, uint8_t status);
    void recvCallback();

private:
    std::shared_ptr<fault_eventPubSubType> m_spFaultEventPub;
    std::shared_ptr<hozon::netaos::cm::Skeleton> m_spSkeleton;

    std::shared_ptr<fault_eventPubSubType> m_spSub;
    std::shared_ptr<hozon::netaos::cm::Proxy> m_spProxy;
    std::shared_ptr<fault_event> m_faultEventData;
};

DiagServerEventPub::DiagServerEventPub()
{
    init();
}

DiagServerEventPub::~DiagServerEventPub()
{
    deInit();
}

void DiagServerEventPub::init()
{
    printf( "DiagServerEventPub::init\n" );
    m_spFaultEventPub = std::make_shared<fault_eventPubSubType>();
    m_spSkeleton = std::make_shared<hozon::netaos::cm::Skeleton>(m_spFaultEventPub);
    m_spSkeleton->Init(0, "fault_eventTopic");

    m_spSub = std::make_shared<fault_eventPubSubType>();
    m_spProxy = std::make_shared<hozon::netaos::cm::Proxy>(m_spSub);
    m_spProxy->Init(0, "fault_eventTopic");
    m_spProxy->Listen(std::bind(&DiagServerEventPub::recvCallback, this));

    m_faultEventData = std::make_shared<fault_event>();
}

void DiagServerEventPub::deInit()
{
    printf( "DiagServerEventPub::deInit\n" );
    m_spProxy->Deinit();
    m_spSkeleton->Deinit();
}

void DiagServerEventPub::sendFaultEvent(uint32_t faultKey, uint8_t status)
{
    timespec curDate;
    clock_gettime(CLOCK_REALTIME, &curDate);
    printf( "DiagServerEventPub::sendFaultEvent cur sec:%ld\n", curDate.tv_sec );

    std::shared_ptr<fault_event> data = std::make_shared<fault_event>();
    data->domain("dtc_recover");
    data->occur_time(curDate.tv_sec);
    data->fault_id(faultKey % 100);
    data->fault_obj((faultKey - faultKey) / 100);
    data->fault_status(status);
    printf( "DiagServerEventPub::sendFaultEvent domain:%s, time:%ld, faultid:%d, faultobj:%d, faultstatus:%d\n",
                data->domain().data(), data->occur_time(), data->fault_id(), (int)data->fault_obj(), (int)data->fault_status() );

    if (m_spSkeleton->IsMatched()) {
        printf("matched\n");
        if (m_spSkeleton->Write(data) == 0) {
            printf("write ok\n");
        }
    }
}

void DiagServerEventPub::recvCallback()
{
    printf( "DiagServerEventPub::recvCallback\n" );
    if (m_spProxy->IsMatched()) {
        m_spProxy->Take(m_faultEventData);
        printf( "DiagServerEventPub::recvCallback domain:%s, time:%ld, faultid:%d, faultobj:%d, faultstatus:%d\n",
                m_faultEventData->domain().data(), m_faultEventData->occur_time(), m_faultEventData->fault_id(),
                (int)m_faultEventData->fault_obj(), (int)m_faultEventData->fault_status() );
    }
    else {
        printf( "DiagServerEventPub::recvCallback m_spProxy not matched\n" );
    }
}
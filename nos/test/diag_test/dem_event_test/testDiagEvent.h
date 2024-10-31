#pragma once

#include "cm/include/skeleton.h"
#include "cm/include/proxy.h"
#include "idl/generated/diagPubSubTypes.h"
#include <memory>
#include <functional>

class TestDiagEvent
{
public:
    TestDiagEvent();
    virtual ~TestDiagEvent();

    void init();
    void deInit();
    void sendFaultEvent(uint32_t iCmd, std::vector<uint8_t>& dataVec);
    void recvCallback();

private:
    std::shared_ptr<testDiagEventPubSubType> m_spFaultEventPub;
    std::shared_ptr<hozon::netaos::cm::Skeleton> m_spSkeleton;

    std::shared_ptr<testDiagEventPubSubType> m_spSub;
    std::shared_ptr<hozon::netaos::cm::Proxy> m_spProxy;
    std::shared_ptr<testDiagEvent> m_testDiagEventData;
};

TestDiagEvent::TestDiagEvent()
{
    init();
}

TestDiagEvent::~TestDiagEvent()
{
    deInit();
}

void TestDiagEvent::init()
{
    printf( "TestDiagEvent::init\n" );
    m_spFaultEventPub = std::make_shared<testDiagEventPubSubType>();
    m_spSkeleton = std::make_shared<hozon::netaos::cm::Skeleton>(m_spFaultEventPub);
    m_spSkeleton->Init(0, "testDiagEvent");

    m_spSub = std::make_shared<testDiagEventPubSubType>();
    m_spProxy = std::make_shared<hozon::netaos::cm::Proxy>(m_spSub);
    m_spProxy->Init(0, "testDiagEvent");
    m_spProxy->Listen(std::bind(&TestDiagEvent::recvCallback, this));

    m_testDiagEventData = std::make_shared<testDiagEvent>();
}

void TestDiagEvent::deInit()
{
    printf( "TestDiagEvent::deInit\n" );
    m_spProxy->Deinit();
    m_spSkeleton->Deinit();
}

void TestDiagEvent::sendFaultEvent(uint32_t iCmd, std::vector<uint8_t>& dataVec)
{
    timespec curDate;
    clock_gettime(CLOCK_REALTIME, &curDate);
    printf( "TestDiagEvent::sendFaultEvent cur sec:%ld\n", curDate.tv_sec );

    std::shared_ptr<testDiagEvent> data = std::make_shared<testDiagEvent>();
    data->iCmd(iCmd);
    data->data_vec(dataVec);

    if (m_spSkeleton->IsMatched()) {
        printf("matched\n");
        if (m_spSkeleton->Write(data) == 0) {
            printf("write ok\n");
        }
    }
}

void TestDiagEvent::recvCallback()
{
    printf( "TestDiagEvent::recvCallback\n" );
    if (m_spProxy->IsMatched()) {
        m_spProxy->Take(m_testDiagEventData);
        printf("icmd:%d\n", m_testDiagEventData->iCmd());
        // printf( "TestDiagEvent::recvCallback domain:%s, time:%ld, faultid:%d, faultobj:%d, faultstatus:%d\n",
        //         m_faultEventData->domain().data(), m_faultEventData->occur_time(), m_faultEventData->fault_id(),
        //         (int)m_faultEventData->fault_obj(), (int)m_faultEventData->fault_status() );
    }
    else {
        printf( "TestDiagEvent::recvCallback m_spProxy not matched\n" );
    }
}
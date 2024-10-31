#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/event_manager/diag_server_event_pub.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerEventPub : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerEventPub* instancepub = nullptr;

TEST_F(TestDiagServerEventPub, DiagServerEventPub)
{
    instancepub = new DiagServerEventPub();
}

TEST_F(TestDiagServerEventPub, deInit)
{
    if (instancepub != nullptr) {
        instancepub->init();
        instancepub->deInit();
    }
}

TEST_F(TestDiagServerEventPub, init)
{
    if (instancepub != nullptr) {
        instancepub->init();
    }
}

TEST_F(TestDiagServerEventPub, sendFaultEvent)
{
    uint32_t faultKey = 800401;
    uint8_t status = 0;
    if (instancepub != nullptr) {
        instancepub->sendFaultEvent(faultKey, status);
    }
}

TEST_F(TestDiagServerEventPub, notifyDtcControlSetting)
{
    uint8_t dtcControlSetting = 0x01;
    if (instancepub != nullptr) {
        instancepub->notifyDtcControlSetting(dtcControlSetting);
    }
}

TEST_F(TestDiagServerEventPub, notifyHmi)
{
    if (instancepub != nullptr) {
        instancepub->notifyHmi();
    }
}
#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/event_manager/diag_server_event_sub.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerEventSub : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerEventSub* instancesub = nullptr;

TEST_F(TestDiagServerEventSub, DiagServerEventSub)
{
    instancesub = new DiagServerEventSub();
}

TEST_F(TestDiagServerEventSub, init)
{
    if (instancesub != nullptr) {
        instancesub->init();
    }
}

TEST_F(TestDiagServerEventSub, registCallback)
{
    if (instancesub != nullptr) {
        instancesub->registCallback();
    }
}

TEST_F(TestDiagServerEventSub, recvCallback)
{
    if (instancesub != nullptr) {
        instancesub->recvCallback();
    }
}
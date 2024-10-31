#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/event_manager/diag_server_event_test.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerEventTest : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerEventTest* instancetest = nullptr;

TEST_F(TestDiagServerEventTest, DiagServerEventTest)
{
    instancetest = new DiagServerEventTest();
}

TEST_F(TestDiagServerEventTest, deInit)
{
    if (instancetest != nullptr) {
        instancetest->init();
        instancetest->deInit();
    }
}

TEST_F(TestDiagServerEventTest, Init)
{
    if (instancetest != nullptr) {
        instancetest->init();
        instancetest->deInit();
    }
}

TEST_F(TestDiagServerEventTest, recvTestCallback)
{
    if (instancetest != nullptr) {
        instancetest->recvTestCallback();

        instancetest->deInit();
        instancetest->recvTestCallback();

    }
}
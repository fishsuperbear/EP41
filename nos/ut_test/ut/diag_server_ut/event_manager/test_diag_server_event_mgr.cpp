#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/event_manager/diag_server_event_mgr.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerEventMgr : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerEventMgr, DiagServerEventMgr)
{
}

TEST_F(TestDiagServerEventMgr, getInstance)
{
}

TEST_F(TestDiagServerEventMgr, DeInit)
{
    DiagServerEventMgr::getInstance()->Init();
    DiagServerEventMgr::getInstance()->DeInit();
}

TEST_F(TestDiagServerEventMgr, Init)
{
    DiagServerEventMgr::getInstance()->Init();
}
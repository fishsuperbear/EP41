#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsBase : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsBase* instanceBase = nullptr;

TEST_F(TestDiagServerUdsBase, DiagServerUdsBase)
{
    instanceBase = new DiagServerUdsBase();
}

TEST_F(TestDiagServerUdsBase, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instanceBase != nullptr) {
        instanceBase->AnalyzeUdsMessage(udsMessage);
    }
}

TEST_F(TestDiagServerUdsBase, PositiveResponse)
{
    DiagServerUdsMessage udsMessage;
    if (instanceBase != nullptr) {
        instanceBase->PositiveResponse(udsMessage);
    }
}

TEST_F(TestDiagServerUdsBase, NegativeResponse)
{
    DiagServerUdsMessage udsMessage;
    if (instanceBase != nullptr) {
        instanceBase->NegativeResponse(udsMessage);
    }
}
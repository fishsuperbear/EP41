#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid85.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid85 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid85* instancesid85 = nullptr;

TEST_F(TestDiagServerUdsSid85, DiagServerUdsSid85)
{
    instancesid85 = new DiagServerUdsSid85();
}

TEST_F(TestDiagServerUdsSid85, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid85 != nullptr) {
        instancesid85->AnalyzeUdsMessage(udsMessage);
    }
}
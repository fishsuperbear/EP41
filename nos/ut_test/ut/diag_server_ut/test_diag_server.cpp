// This is googletest template, please change TestCaseName and TestName, write you own test code here.

#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/diag_server.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;
// using namespace hozon::netaos::diag::cm_transport;
class TestDiagServer : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServer* instance = nullptr;

TEST_F(TestDiagServer, DiagServer)
{
    instance = new DiagServer();
}

TEST_F(TestDiagServer, DeInit)
{
    if (instance != nullptr) {
        // instance->Init();
        instance->DeInit();
    }
}

TEST_F(TestDiagServer, Init)
{
    if (instance != nullptr) {
        instance->Init();
    }
}

TEST_F(TestDiagServer, Stop)
{
    if (instance != nullptr) {
        instance->Stop();
    }
}

TEST_F(TestDiagServer, Run)
{
    if (instance != nullptr) {
        instance->Stop();
        instance->Run();
        delete instance;
        instance = nullptr;
    }
}
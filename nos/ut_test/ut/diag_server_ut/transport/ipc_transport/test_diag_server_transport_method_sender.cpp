#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/diag_server.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerConfig : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};
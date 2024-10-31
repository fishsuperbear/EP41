#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/event_manager/diag_server_event_status.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerEventStatus : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerEventStatus* instancestatus = nullptr;

TEST_F(TestDiagServerEventStatus, DiagServerEventStatus)
{
    instancestatus = new DiagServerEventStatus();
}

TEST_F(TestDiagServerEventStatus, setTripCounter)
{
    uint8_t iTripCounter = 0x01;
    if (instancestatus != nullptr) {
        instancestatus->setTripCounter(iTripCounter);
    }
}

TEST_F(TestDiagServerEventStatus, setDTCStatus)
{
    uint8_t iTripCounter = 0x01;
    if (instancestatus != nullptr) {
        instancestatus->setDTCStatus(iTripCounter);
    }
}

TEST_F(TestDiagServerEventStatus, getDTCStatus)
{
    // uint8_t iTripCounter = 0x01;
    if (instancestatus != nullptr) {
        instancestatus->getDTCStatus();
    }
}

TEST_F(TestDiagServerEventStatus, setStatusSetting)
{
    DIAG_CONTROLDTCSTATUSTYPE bStatusSetting = DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOn;
    if (instancestatus != nullptr) {
        instancestatus->setStatusSetting(bStatusSetting);
    }
}

TEST_F(TestDiagServerEventStatus, getStatusSetting)
{
    // uint8_t iTripCounter = 0x01;
    if (instancestatus != nullptr) {
        instancestatus->getStatusSetting();
    }
}

TEST_F(TestDiagServerEventStatus, onStatusChange)
{
    DIAG_DTCSTSCHGCON eStaChaCon = DIAG_DTCSTSCHGCON::DIAG_DTCSTSCHGCON_OCCUR;
    if (instancestatus != nullptr) {
        instancestatus->onStatusChange(eStaChaCon);
    }
}

// TEST_F(TestDiagServerEventStatus, dealWithDTCChange)
// {
//     DIAG_DTCSTSCHGCON eStaChaCon;
//     if (instancestatus != nullptr) {
//         instancestatus->dealWithDTCChange(eStaChaCon);
//     }
// }

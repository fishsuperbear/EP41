#include <iostream>
#include <crc.h>
#include "gtest/gtest.h"
#include "e2exf_impl.h"

#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <string>
#include <thread>
#include <random>
#include <chrono>

using namespace std;
using namespace hozon::netaos::e2e;

sig_atomic_t g_stopFlag = 0;
std::mt19937 gen(std::random_device{}());
std::uniform_int_distribution<uint8_t> dist(0,255);

class E2EFuncTest : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {

    }
    static void TearDownTestSuite() {

    }
    void SetUp() override {
        SMConfig.ClearToInvalid = false;
        SMConfig.transitToInvalidExtended = true;
        SMConfig.WindowSizeValid = 10;
        SMConfig.WindowSizeInvalid = 10;
        SMConfig.WindowSizeInit = 10;
        SMConfig.MaxErrorStateInit = 5;
        SMConfig.MinOkStateInit = 1;
        SMConfig.MaxErrorStateInvalid = 5;
        SMConfig.MinOkStateInvalid = 1;
        SMConfig.MaxErrorStateValid = 5;
        SMConfig.MinOkStateValid = 1;
    }
    void TearDown() override {}

 protected:
    E2E_SMConfigType SMConfig;
    E2EXf_ConfigType Config;
    uint8_t DataIDList[16] = {174,124,106,58,47,154,237,220,98,37,173,212,59,125,101,188};
};

TEST_F(E2EFuncTest, Profile22Custom) {
    for (int i = 0; i < 16; i++) 
        Config.ProfileConfig.Profile22.DataIDList[i] = DataIDList[i];
    Config.ProfileConfig.Profile22.DataLength = 33 * 8;
    Config.ProfileConfig.Profile22.MaxDeltaCounter = 4;
    Config.ProfileConfig.Profile22.Offset = 0;

    Config.Profile = E2EXf_Profile::PROFILE22_CUSTOM;
    Config.disableEndToEndCheck = FALSE;
    Config.disableEndToEndStatemachine = FALSE;
    Config.headerLength = 0 * 8;
    Config.InPlace = TRUE;
    Config.upperHeaderBitsToShift = 0 * 8;
    Config.DataTransformationStatusForwarding = noTransformerStatusForwarding;

    E2EXf_Index index(Config.ProfileConfig.Profile22.DataIDList, /*canmsg id*/0x386);

    E2EXf_Config config(Config, SMConfig);

    AddE2EXfConfig(index, config);

    uint32_t dataLength = 32;
    Payload data(dataLength);
    for (size_t i = 0; i < dataLength; i++) 
        data[i] = dist(gen);
    for (uint32_t loop = 0; loop < 10000; loop++) {
        {
            ProtectResult Result = E2EXf_Protect(index, data, data.size());
            GTEST_ASSERT_EQ(ProtectResult::E_OK, Result);
        }
        {
            CheckResult Result = E2EXf_Check(index, data, data.size());
            GTEST_ASSERT_EQ(E2EXf_PCheckStatusType::E2E_P_OK, Result.GetProfileCheckStatus());
        }
    }
}

TEST_F(E2EFuncTest, Profile22) {
    for (int i = 0; i < 16; i++) 
        Config.ProfileConfig.Profile22.DataIDList[i] = DataIDList[i];
    Config.ProfileConfig.Profile22.DataLength = 32 * 8;
    Config.ProfileConfig.Profile22.MaxDeltaCounter = 4;
    Config.ProfileConfig.Profile22.Offset = 0;

    Config.Profile = E2EXf_Profile::PROFILE22;
    Config.disableEndToEndCheck = FALSE;
    Config.disableEndToEndStatemachine = FALSE;
    Config.headerLength = 0 * 8;
    Config.InPlace = TRUE;
    Config.upperHeaderBitsToShift = 0 * 8;
    Config.DataTransformationStatusForwarding = noTransformerStatusForwarding;

    E2EXf_Index index(Config.ProfileConfig.Profile22.DataIDList, /*canmsg id*/0x386);

    E2EXf_Config config(Config, SMConfig);

    AddE2EXfConfig(index, config);

    uint32_t dataLength = 32;
    Payload data(dataLength);
    for (uint32_t loop = 0; loop < 10000; loop++) {
        for (size_t i = 0; i < dataLength; i++) 
            data[i] = dist(gen);
        {
            ProtectResult Result = E2EXf_Protect(index, data, data.size());
            GTEST_ASSERT_EQ(ProtectResult::E_OK, Result);
        }
        {
            CheckResult Result = E2EXf_Check(index, data, data.size());
            GTEST_ASSERT_EQ(E2EXf_PCheckStatusType::E2E_P_OK, Result.GetProfileCheckStatus());
        }
    }
}

TEST_F(E2EFuncTest, Profile04) {

    Config.ProfileConfig.Profile04.DataID = 123123;
    Config.ProfileConfig.Profile04.MinDataLength = 0;
    Config.ProfileConfig.Profile04.MaxDataLength = 32 * 8;
    Config.ProfileConfig.Profile04.MaxDeltaCounter = 4;
    Config.ProfileConfig.Profile04.Offset = 0;

    Config.Profile = E2EXf_Profile::PROFILE04;
    Config.disableEndToEndCheck = FALSE;
    Config.disableEndToEndStatemachine = FALSE;
    Config.headerLength = 0 * 8;
    Config.InPlace = TRUE;
    Config.upperHeaderBitsToShift = 0 * 8;
    Config.DataTransformationStatusForwarding = noTransformerStatusForwarding;

    E2EXf_Index index(Config.ProfileConfig.Profile04.DataID, /*canmsg id*/0x386);

    E2EXf_Config config(Config, SMConfig);

    AddE2EXfConfig(index, config);

    uint32_t dataLength = 32;
    Payload data(dataLength);
    for (size_t i = 0; i < dataLength; i++) 
        data[i] = dist(gen);
    for (uint32_t loop = 0; loop < 10000; loop++) {
        {
            ProtectResult Result = E2EXf_Protect(index, data, data.size());
            GTEST_ASSERT_EQ(ProtectResult::E_OK, Result);
        }
        {
            CheckResult Result = E2EXf_Check(index, data, data.size());
            GTEST_ASSERT_EQ(E2EXf_PCheckStatusType::E2E_P_OK, Result.GetProfileCheckStatus());
        }
    }
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
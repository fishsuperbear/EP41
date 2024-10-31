#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/info/diag_server_dynamic_info.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerDynamicInfo : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerDynamicInfo, DiagServerDynamicInfo)
{
}

TEST_F(TestDiagServerDynamicInfo, getInstance)
{
}

TEST_F(TestDiagServerDynamicInfo, DeInit)
{
    DiagServerDynamicInfo::getInstance()->Init();
    DiagServerDynamicInfo::getInstance()->DeInit();
}

TEST_F(TestDiagServerDynamicInfo, Init)
{
    DiagServerDynamicInfo::getInstance()->Init();
}

TEST_F(TestDiagServerDynamicInfo, ReadInstallStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadInstallStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadPowerSupplyVoltage)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadPowerSupplyVoltage(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadOdometerValue)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadOdometerValue(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadVehicleSpeed)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadVehicleSpeed(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadIgnitionStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadIgnitionStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadTime)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadTime(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadPKIApplyStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadPKIApplyStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASF30CameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASF30CameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASF120CameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASF120CameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASFLCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASFLCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASFRCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASFRCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASRLCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASRLCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASRRCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASRRCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASRearCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASRearCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASF30AndF120CameraCoordinatedCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASF30AndF120CameraCoordinatedCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASF120AndRLCameraCoordinatedCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASF120AndRLCameraCoordinatedCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASF120AndRRCameraCoordinatedCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASF120AndRRCameraCoordinatedCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASFLAndRLCameraCoordinatedCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASFLAndRLCameraCoordinatedCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASFRAndRRCameraCoordinatedCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASFRAndRRCameraCoordinatedCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASFLAndRearCameraCoordinatedCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASFLAndRearCameraCoordinatedCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadADASFRAndRearCameraCoordinatedCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadADASFRAndRearCameraCoordinatedCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadAfterSalesADASF30CameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASF30CameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadAfterSalesADASF120CameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASF120CameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadAfterSalesADASFLCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASFLCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadAfterSalesADASFRCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASFRCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadAfterSalesADASRLCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASRLCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadAfterSalesADASRRCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASRRCameraCalibrationStatus(data);
}

TEST_F(TestDiagServerDynamicInfo, ReadAfterSalesADASRearCameraCalibrationStatus)
{
    std::vector<uint8_t> data;
    DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASRearCameraCalibrationStatus(data);
}
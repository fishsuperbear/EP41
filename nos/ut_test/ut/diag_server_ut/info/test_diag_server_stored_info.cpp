#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/info/diag_server_stored_info.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerStoredInfo : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerStoredInfo, DiagServerStoredInfo)
{
}

TEST_F(TestDiagServerStoredInfo, getInstance)
{
}

TEST_F(TestDiagServerStoredInfo, DeInit)
{
    DiagServerStoredInfo::getInstance()->Init();
    DiagServerStoredInfo::getInstance()->DeInit();
}

TEST_F(TestDiagServerStoredInfo, Init)
{
    DiagServerStoredInfo::getInstance()->Init();
}

TEST_F(TestDiagServerStoredInfo, WriteVehicleCfgWordData)
{
    std::vector<uint8_t> data;
    for (int i = 0; i < 58; ++i) {
        data.push_back(static_cast<uint8_t>(0x2f));
    }
    DiagServerStoredInfo::getInstance()->WriteVehicleCfgWordData(data);
}

TEST_F(TestDiagServerStoredInfo, ReadVehicleCfgWordData)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadVehicleCfgWordData(data);
}

TEST_F(TestDiagServerStoredInfo, WriteVinData)
{
    std::vector<uint8_t> data;
    for (int i = 0; i < 17; ++i) {
        data.push_back(static_cast<uint8_t>(0x2f));
    }
    DiagServerStoredInfo::getInstance()->WriteVinData(data);
}

TEST_F(TestDiagServerStoredInfo, ReadVinData)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadVinData(data);
}

TEST_F(TestDiagServerStoredInfo, ReadECUSWData)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadECUSWData(data);
}

TEST_F(TestDiagServerStoredInfo, WriteTesterSNData)
{
    std::vector<uint8_t> data;
    for (int i = 0; i < 10; ++i) {
        data.push_back(static_cast<uint8_t>(0x2f));
    }
    DiagServerStoredInfo::getInstance()->WriteTesterSNData(data);
}

TEST_F(TestDiagServerStoredInfo, ReadTesterSNData)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadTesterSNData(data);
}

TEST_F(TestDiagServerStoredInfo, WriteProgrammingDateData)
{
    std::vector<uint8_t> data;
    for (int i = 0; i < 4; ++i) {
        data.push_back(static_cast<uint8_t>(0x2f));
    }
    DiagServerStoredInfo::getInstance()->WriteProgrammingDateData(data);
}

TEST_F(TestDiagServerStoredInfo, ReadProgrammingDateData)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadProgrammingDateData(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEcuType)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEcuType(data);
}

TEST_F(TestDiagServerStoredInfo, WriteInstallDate)
{
    std::vector<uint8_t> data;
    for (int i = 0; i < 4; ++i) {
        data.push_back(static_cast<uint8_t>(0x2f));
    }
    DiagServerStoredInfo::getInstance()->WriteInstallDate(data);
}

TEST_F(TestDiagServerStoredInfo, ReadInstallDate)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadInstallDate(data);
}

TEST_F(TestDiagServerStoredInfo, WriteEskNumber)
{
    std::vector<uint8_t> data;
    for (int i = 0; i < 16; ++i) {
        data.push_back(static_cast<uint8_t>(0x2f));
    }
    DiagServerStoredInfo::getInstance()->WriteEskNumber(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEskNumber)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEskNumber(data);
}

TEST_F(TestDiagServerStoredInfo, ReadBootSWId)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadBootSWId(data);
}

TEST_F(TestDiagServerStoredInfo, ReadCurrDiagSession)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadCurrDiagSession(data);
}

TEST_F(TestDiagServerStoredInfo, ReadVehicleManufacturerSparePartNumber)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadVehicleManufacturerSparePartNumber(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEcuSoftwareNumber)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEcuSoftwareNumber(data);
}

TEST_F(TestDiagServerStoredInfo, ReadSystemSupplierId)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadSystemSupplierId(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEcuManufactureDate)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEcuManufactureDate(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEcuSerialNumber)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEcuSerialNumber(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEcuHardwareVersion)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEcuHardwareVersion(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEcuHardwareNumber)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEcuHardwareNumber(data);
}

TEST_F(TestDiagServerStoredInfo, ReadEcuSoftwareAssemblyPartNumber)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadEcuSoftwareAssemblyPartNumber(data);
}

TEST_F(TestDiagServerStoredInfo, ReadAllSensorVersions)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadAllSensorVersions(data);
}

TEST_F(TestDiagServerStoredInfo, ReadOrinVersion)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadOrinVersion(data);
}

TEST_F(TestDiagServerStoredInfo, ReadSOCVersion)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadSOCVersion(data);
}

TEST_F(TestDiagServerStoredInfo, ReadMCUVersion)
{
    std::vector<uint8_t> data;
    DiagServerStoredInfo::getInstance()->ReadMCUVersion(data);
}
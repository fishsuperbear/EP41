#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/datatransfer/diag_server_data_transfer.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerDataTransfer : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerDataTransfer, DiagServerDataTransfer)
{
}

TEST_F(TestDiagServerDataTransfer, getInstance)
{
}

TEST_F(TestDiagServerDataTransfer, DeInit)
{
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->DeInit();
}

TEST_F(TestDiagServerDataTransfer, Init)
{
    DiagServerDataTransfer::getInstance()->Init();
}

TEST_F(TestDiagServerDataTransfer, StartFileDataUpload)
{
    std::string path = "";
    DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);

    path = "/app/version.jso";
    DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);

    path = "/app/test/sample/log_sample/conf";
    DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);

    path = "/app/version.json";
    DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);
}

TEST_F(TestDiagServerDataTransfer, StopFileDataUpload)
{
    // DiagServerDataTransfer::getInstance()->StopFileDataUpload();
}

TEST_F(TestDiagServerDataTransfer, StartFileDataDownload)
{
    std::string path = "";
    uint64_t size = 0;
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);

    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);

    path = "/app/version.jso";
    size = 1;
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);

    path = "/app/test/sample/log_sample/conf";
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);

    path = "/app/version.json";
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);
}

TEST_F(TestDiagServerDataTransfer, StopFileDataDownload)
{
    // DiagServerDataTransfer::getInstance()->StopFileDataDownload();
}

TEST_F(TestDiagServerDataTransfer, StopDataTransfer)
{
    DiagServerDataTransfer::getInstance()->StopDataTransfer();
}

TEST_F(TestDiagServerDataTransfer, ReadDataBlockByCounter)
{
    uint8_t serverId = 0x00;
    std::vector<uint8_t> vec;
    DiagServerDataTransfer::getInstance()->ReadDataBlockByCounter(serverId, vec);

    DiagServerDataTransfer::getInstance()->StopDataTransfer();
    DiagServerDataTransfer::getInstance()->ReadDataBlockByCounter(serverId, vec);

    std::string path = "/app/version.json";
    uint64_t size = 0;
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);
    DiagServerDataTransfer::getInstance()->ReadDataBlockByCounter(serverId, vec);

    size = 1;
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);
    DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);
    DiagServerDataTransfer::getInstance()->ReadDataBlockByCounter(serverId, vec);

    // path = "/app/test/sample/log_sample/conf";
    // DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);
    // DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);
    // DiagServerDataTransfer::getInstance()->ReadDataBlockByCounter(serverId, vec);
}

TEST_F(TestDiagServerDataTransfer, WriteDataToFileByCounter)
{
    uint8_t serverId = 0x00;
    std::vector<uint8_t> vec;
    DiagServerDataTransfer::getInstance()->WriteDataToFileByCounter(serverId, vec);

    DiagServerDataTransfer::getInstance()->StopDataTransfer();
    DiagServerDataTransfer::getInstance()->WriteDataToFileByCounter(serverId, vec);

    std::string path = "/app/version.jso";
    uint64_t size = 0;
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);
    DiagServerDataTransfer::getInstance()->WriteDataToFileByCounter(serverId, vec);

    size = 1;
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->StartFileDataDownload(path, size);
    // DiagServerDataTransfer::getInstance()->StartFileDataUpload(path);
    DiagServerDataTransfer::getInstance()->WriteDataToFileByCounter(serverId, vec);

}

TEST_F(TestDiagServerDataTransfer, GetFilesInfo)
{
    // uint8_t serverId = 0x01;
    // DiagServerDataTransfer::getInstance()->GetFilesInfo();
}

TEST_F(TestDiagServerDataTransfer, GetSizeToVecWithType)
{
    // uint8_t serverId = 0x01;
    std::vector<uint8_t> vec;
    DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTranferSizeType::TRANSCAPACITY, vec);
}
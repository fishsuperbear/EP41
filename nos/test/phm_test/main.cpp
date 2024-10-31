#include <sys/time.h>
#include <signal.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <unistd.h>
#include "log/include/logging.h"
#include "phm/include/phm_client.h"

using namespace hozon::netaos::phm;

uint8_t stopFlag = 0;
struct timeval tStart, tEnd;
std::shared_ptr<PHMClient> spPHMClient = nullptr;

void SigHandler(int signum)
{
    std::cout << "PhmTest sigHandler enter, signum:" << signum << std::endl;
    stopFlag = 1;
}

void ServiceAvailableCallback(bool bResult)
{
    std::cout << "PhmTest ServiceAvailableCallback bResult: " << bResult << std::endl;
}

void FaultReceiveCallback(const ReceiveFault_t& fault)
{
    std::cout << "PhmTest FaultReceiveCallback faultId: " << (int)fault.faultId
              << " faultObj: " << static_cast<uint>(fault.faultObj)
              << " faultStatus: " << (int)fault.faultStatus
              << " faultOccurTime: " << fault.faultOccurTime
              << " faultDomain: " << fault.faultDomain
              << " faultDes: " << fault.faultDes
              << "---------------------------------------\n";

    for (auto& item : fault.faultCluster) {
        std::cout << "PhmTest FaultReceiveCallback clusterName: " << item.clusterName
                  << " clusterLevel: " << (int)item.clusterValue << std::endl;
    }
}

void start()
{
    gettimeofday(&tStart,NULL);
}

void end()
{
    gettimeofday(&tEnd,NULL);
    int milsec = (tEnd.tv_sec*1000 + tEnd.tv_usec/1000) - (tStart.tv_sec*1000 + tStart.tv_usec/1000);
    printf("Test over! cost time: %d ms\n", milsec);
}

void TestInit()
{
    // test has/not has config.yaml
    // spPHMClient->Init("/app/conf/phm_config.yaml", ServiceAvailableCallback, FaultReceiveCallback);
    // spPHMClient->Init("", ServiceAvailableCallback, FaultReceiveCallback);
    spPHMClient->Init();
}

void TestStart()
{
    printf("TestStart\n");
    // spPHMClient->Start();
    spPHMClient->Start(100);
}

void TestStop()
{
    printf("TestStop\n");
    spPHMClient->Stop();
}

void TestDeinit()
{
    printf("TestDeinit\n");
    spPHMClient->Deinit();
}

void TestInhibitFault()
{
    printf("TestInhibitFault\n");
    int faultId1 = 8128;
    int faultObj1 = 1;
    int faultId2 = 8004;
    int faultObj2 = 1;
    std::vector<uint32_t> faultKeys;
    faultKeys.emplace_back(faultId1*100 + faultObj1);
    faultKeys.emplace_back(faultId2*100 + faultObj2);
    spPHMClient->InhibitFault(faultKeys);
}

void TestRecoverInhibitFault()
{
    printf("TestRecoverInhibitFault\n");
    int faultId1 = 8128;
    int faultObj1 = 1;
    int faultId2 = 8004;
    int faultObj2 = 1;
    std::vector<uint32_t> faultKeys;
    faultKeys.emplace_back(faultId1*100 + faultObj1);
    faultKeys.emplace_back(faultId2*100 + faultObj2);
    spPHMClient->RecoverInhibitFault(faultKeys);
}

void TestInhibitAllFault()
{
    printf("TestInhibitAllFault\n");
    spPHMClient->InhibitAllFault();
}

void TestRecoverInhibitAllFault()
{
    printf("TestRecoverInhibitAllFault\n");
    spPHMClient->RecoverInhibitAllFault();
}

void TestGetDataCollectionFile()
{
    printf("TestGetDataCollectionFile\n");
    std::vector<std::string> outResult;
    spPHMClient->GetDataCollectionFile(outResult);
    for (auto& file : outResult) {
        printf("GetDataCollectionFile:%s\n", file.data());
    }
}

void ReportFault()
{
    printf("ReportFault\n");
    // const SendFault_t sendFault(8005, 1, 1);
    const SendFault_t sendFault(4000, 26, 1);
    spPHMClient->ReportFault(sendFault);
}

void ReportFaultWithCount()
{
    printf("ReportFaultWithCount\n");
    SendFault_t sendFault(2000, 1, 1);
    sendFault.faultDebounce.debounceType = hozon::netaos::phm::DebounceType::DEBOUNCE_TYPE_COUNT;
    sendFault.faultDebounce.debounceSetting.countDebounce.debounceCount = 0;
    sendFault.faultDebounce.debounceSetting.countDebounce.debounceTime = 0;
    spPHMClient->ReportFault(sendFault);
    spPHMClient->ReportFault(sendFault);
    // spPHMClient->ReportFault(sendFault);
}

void ReportFaultWithTime()
{
    printf("ReportFaultWithTime\n");
    SendFault_t sendFault(2000, 2, 1);
    sendFault.faultDebounce.debounceType = hozon::netaos::phm::DebounceType::DEBOUNCE_TYPE_TIME;
    sendFault.faultDebounce.debounceSetting.countDebounce.debounceTime = 0;
    spPHMClient->ReportFault(sendFault);
}


void ReportCheckPoint()
{
    printf("ReportCheckPoint 0\n");
    uint32_t checkPointId = 0;
    spPHMClient->ReportCheckPoint(checkPointId);
}

std::thread MultiThreadReportThd1;
std::thread MultiThreadReportThd2;
std::thread MultiThreadReportThd3;
std::thread MultiThreadReportThd4;

void MultiThreadReportTest()
{
    // INFO_LOG << "#### PhmFuncTest::MultiThreadReportTest Starting...";
    MultiThreadReportThd1 = std::thread([]{
        std::shared_ptr<PHMClient> spPHMClient = nullptr;
        spPHMClient= std::make_shared<hozon::netaos::phm::PHMClient>();

        spPHMClient->Init();

        hozon::netaos::phm::SendFault_t cSendFault(4020, 5, 1);
        while (1) {
            cSendFault.faultStatus = 1;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            cSendFault.faultStatus = 0;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    MultiThreadReportThd1.detach();

    MultiThreadReportThd2 = std::thread ([]{
        std::shared_ptr<PHMClient> spPHMClient = nullptr;
        spPHMClient= std::make_shared<hozon::netaos::phm::PHMClient>();

        spPHMClient->Init();

        hozon::netaos::phm::SendFault_t cSendFault(4020, 6, 1);
        while (1) {
            cSendFault.faultStatus = 1;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            cSendFault.faultStatus = 0;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    MultiThreadReportThd2.detach();

    MultiThreadReportThd3 = std::thread ([]{
        std::shared_ptr<PHMClient> spPHMClient = nullptr;
        spPHMClient= std::make_shared<hozon::netaos::phm::PHMClient>();

        spPHMClient->Init();

        hozon::netaos::phm::SendFault_t cSendFault(6666, 6, 1);
        while (1) {
            cSendFault.faultStatus = 1;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            cSendFault.faultStatus = 0;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    MultiThreadReportThd3.detach();

    MultiThreadReportThd4 = std::thread ([]{
        std::shared_ptr<PHMClient> spPHMClient = nullptr;
        spPHMClient= std::make_shared<hozon::netaos::phm::PHMClient>();

        spPHMClient->Init();

        hozon::netaos::phm::SendFault_t cSendFault(4010, 26, 1);
        cSendFault.faultDebounce.debounceType = hozon::netaos::phm::DebounceType::DEBOUNCE_TYPE_TIME;
        cSendFault.faultDebounce.debounceSetting.countDebounce.debounceCount = 10;
        cSendFault.faultDebounce.debounceSetting.countDebounce.debounceTime = 1000;
        cSendFault.faultDebounce.debounceSetting.timeDebounce.debounceTime = 1000;

        while (1) {
            cSendFault.faultStatus = 1;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            cSendFault.faultStatus = 0;
            spPHMClient->ReportFault(cSendFault);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    MultiThreadReportThd4.detach();
}


int main(int argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    hozon::netaos::log::InitLogging("phm_test", "phm_test",
                                    hozon::netaos::log::LogLevel::kDebug,
                                    hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
                                    "./", 10, 100);

#if 0
    typedef void(*pFunc)();
    std::vector<std::vector<pFunc> > vp
    {
        { TestInit },
        { TestStart },
        { TestStop },
        { TestDeinit },
        { TestInhibitFault },
        { TestRecoverInhibitFault },
        { TestInhibitAllFault },
        { TestRecoverInhibitAllFault },

        { TestInit,TestStart },
        { TestInit,TestStop },
        { TestInit,TestDeinit },

        { TestStart, TestInit },
        { TestStart, TestStop },
        { TestStart, TestDeinit },

        { TestStop, TestInit },
        { TestStop, TestStart },
        { TestStop, TestDeinit },
        { TestStop, TestInhibitFault },
        { TestStop, TestRecoverInhibitFault },
        { TestStop, TestInhibitAllFault },
        { TestStop, TestRecoverInhibitAllFault },

        { TestDeinit,TestInit },
        { TestDeinit,TestStart },
        { TestDeinit,TestStop },

        { TestInit, TestStop, TestInit },
        { TestInit, TestStop, TestStart },
        { TestInit, TestStop, TestDeinit },
        { TestInit, TestStop, TestInhibitFault },
        { TestInit, TestStop, TestRecoverInhibitFault },
        { TestInit, TestStop, TestInhibitAllFault },
        { TestInit, TestStop, TestRecoverInhibitAllFault },
    };
    for (size_t i = 0; i < vp.size(); ++i) {
        spPHMClient.reset(new PHMClient());
        for (size_t j = 0; j < vp[i].size(); ++j) {
            start();
            vp[i][j]();
            end();
        }
        spPHMClient.reset();
        printf("\n\n");
    }


    {
        // test GetDataCollectionFile
        spPHMClient.reset(new PHMClient());
        start();
        TestGetDataCollectionFile();
        // TestGetDataCollectionFile();
        end();
        printf("TestGetDataCollectionFile end\n\n");

        // test report fault
        spPHMClient.reset(new PHMClient());
        TestInit();
        sleep(6);
        start();
        ReportFault();
        end();

        ReportFaultWithCount();
        ReportFaultWithTime();
        printf("\n\n");
    }


    {
        // test alive
        spPHMClient.reset(new PHMClient());
        TestInit();
        start();
        TestStart(); // dealy 100ms
        end();
        std::this_thread::sleep_for(std::chrono::milliseconds(101));
        int count = 10;
        while(count-- > 0) {
            printf("count:%d\n", count);
            ReportCheckPoint();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10000));
        count = 20;
        while(count-- > 0) {
            printf("count:%d\n", count);
            ReportCheckPoint();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        TestStop();
    }


    {
        // test deadline
        spPHMClient.reset(new PHMClient());
        TestInit();
        sleep(6);
        start();
        TestStart();
        end();

        uint32_t checkPointIdS = 21;
        uint32_t checkPointIdE = 22;
        {
            // less than
            start();
            spPHMClient->ReportCheckPoint(checkPointIdS);
            end();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            start();
            spPHMClient->ReportCheckPoint(checkPointIdE);
            end();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            // recover
            spPHMClient->ReportCheckPoint(checkPointIdS);
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            spPHMClient->ReportCheckPoint(checkPointIdE);
        }

        {
            // equal 1
            spPHMClient->ReportCheckPoint(checkPointIdS);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            spPHMClient->ReportCheckPoint(checkPointIdE);
        }

        {
            // equal 2
            spPHMClient->ReportCheckPoint(checkPointIdS);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            spPHMClient->ReportCheckPoint(checkPointIdE);
        }

        {
            // equal 3
            spPHMClient->ReportCheckPoint(checkPointIdS);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            spPHMClient->ReportCheckPoint(checkPointIdE);
        }

        {
            // more than
            spPHMClient->ReportCheckPoint(checkPointIdS);
            std::this_thread::sleep_for(std::chrono::milliseconds(510));
            spPHMClient->ReportCheckPoint(checkPointIdE);
        }
    }

#endif


    // MultiThreadReportTest();

    spPHMClient.reset(new PHMClient());
    TestInit();
    // sleep(60);
    ReportFaultWithCount();
    ReportFaultWithTime();

    // // regiter cluster test "r-camera"
    // const SendFault_t sendFault1(4100, 1, 1);
    // spPHMClient->ReportFault(sendFault1);
    // sleep(2);

    // // regiter fault test
    // const SendFault_t sendFault2(4100, 2, 1);
    // spPHMClient->ReportFault(sendFault2);
    // sleep(2);

    // hozon::netaos::phm::SendFault_t cSendFault(4100, 6, 1);
    // while (1) {
    //     cSendFault.faultStatus = 1;
    //     spPHMClient->ReportFault(cSendFault);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));

    //     cSendFault.faultStatus = 0;
    //     spPHMClient->ReportFault(cSendFault);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }

    // const SendFault_t sendFault21(4000, 14, 1);
    // spPHMClient->ReportFault(sendFault21);
    // sleep(2);

    // // regiter combination test
    // const SendFault_t sendFault3(4700, 1, 1);
    // spPHMClient->ReportFault(sendFault3);
    // sleep(2);

    // const SendFault_t sendFault4(4000, 26, 0);
    // spPHMClient->ReportFault(sendFault4);
    // sleep(2);

    // const SendFault_t sendFault5(4100, 1, 0);
    // spPHMClient->ReportFault(sendFault5);
    // sleep(2);

    // const SendFault_t sendFault51(4000, 14, 0);
    // spPHMClient->ReportFault(sendFault51);
    // sleep(2);

    // const SendFault_t sendFault6(4700, 1, 0);
    // spPHMClient->ReportFault(sendFault6);

    while (!stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20000));
    }

    // spPHMClient->Stop();
    // spPHMClient->Deinit();
    return 0;
}

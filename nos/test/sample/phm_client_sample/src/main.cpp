#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>

#include "phm_client_instance.h"
#include "log/include/logging.h"

bool stopFlag = false;
const std::string REGEX = " ";

void SigHandler(int signum)
{
    std::cout << "--- phm client sample sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = true;
}

void ReportFaultSample()
{
    // report once
    int faultId = 8128;
    int faultObj = 1;
    int faultStatus = 1;
    SendFault_t sendOnceFault(faultId, faultObj, faultStatus);
    PhmClientInstance::getInstance()->ReportFault(sendOnceFault);

    // report periodic(1000ms)
    int periodTime = 1000;
    std::thread thread_report = std::thread([=](){
        int faultId = 8128;
        int faultObj = 1;
        int faultStatus = 1;
        while(1) {
            if (stopFlag) {
                break;
            }

            if (0 == faultStatus) {
                faultStatus = 1;
                SendFault_t sendPeriodFault(faultId, faultObj, faultStatus);
                PhmClientInstance::getInstance()->ReportFault(sendPeriodFault);
            }
            else {
                faultStatus = 0;
                SendFault_t sendPeriodFault(faultId, faultObj, faultStatus);
                PhmClientInstance::getInstance()->ReportFault(sendPeriodFault);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(periodTime));
        }
    });

    thread_report.detach();
}

void ReportFaultRecoverySample()
{
    // report once
    uint32_t faultId = 4000;
    uint8_t faultObj = 9;
    uint8_t faultStatus = 1;
    bool isAutoRecovery = true;
    uint32_t autoRecoveryTime = 5000;
    SendFault_t sendOnceFault(faultId, faultObj, faultStatus, isAutoRecovery, autoRecoveryTime);
    PhmClientInstance::getInstance()->ReportFault(sendOnceFault);

    // report periodic(1000ms)
    int periodTime = 1000;
    std::thread thread_report = std::thread([=](){
        uint32_t faultId = 4000;
        uint8_t faultObj = 8;
        uint8_t faultStatus = 1;
        bool isAutoRecovery = true;
        uint32_t autoRecoveryTime = 3000;
        int i = 0;
        while(i <= 4) {
            if (stopFlag) {
                break;
            }

            SendFault_t sendOnceFault(faultId, faultObj, faultStatus, isAutoRecovery, autoRecoveryTime);
            PhmClientInstance::getInstance()->ReportFault(sendOnceFault);
            ++i;
            std::this_thread::sleep_for(std::chrono::milliseconds(periodTime));
        }
    });

    thread_report.detach();
}

void AliveTaskSample()
{
    int checkPointId = 0;
    int periodTime = 1000;
    std::thread thread_task = std::thread([=](){
        while(1) {
            if (stopFlag) {
                break;
            }

            PhmClientInstance::getInstance()->ReportCheckPoint(checkPointId);
            std::this_thread::sleep_for(std::chrono::milliseconds(periodTime));
        }
    });

    thread_task.detach();
}

void DeadLineTaskSample()
{
    std::vector<int> checkPointId = {1, 2};
    int deadlineMinMs = 50;
    int deadlineMaxMs = 500;
    for (auto& pointId : checkPointId) {
        PhmClientInstance::getInstance()->ReportCheckPoint(pointId);
        std::this_thread::sleep_for(std::chrono::milliseconds(deadlineMaxMs - deadlineMinMs));
    }
}

void LogicTaskSample()
{
    std::vector<int> checkPointId = {3, 4, 5};
    for (auto& pointId : checkPointId) {
        PhmClientInstance::getInstance()->ReportCheckPoint(pointId);
    }
}

void InhibitFaultSample()
{
    int faultId1 = 8128;
    int faultObj1 = 1;
    int faultId2 = 8004;
    int faultObj2 = 1;
    std::vector<uint32_t> faultKeys;
    faultKeys.emplace_back(faultId1*100 + faultObj1);
    faultKeys.emplace_back(faultId2*100 + faultObj2);
    PhmClientInstance::getInstance()->InhibitFault(faultKeys);
}

void RecoverFaultSample()
{
    int faultId1 = 8128;
    int faultObj1 = 1;
    int faultId2 = 8004;
    int faultObj2 = 1;
    std::vector<uint32_t> faultKeys;
    faultKeys.emplace_back(faultId1*100 + faultObj1);
    faultKeys.emplace_back(faultId2*100 + faultObj2);
    PhmClientInstance::getInstance()->RecoverInhibitFault(faultKeys);
}

void InhibitAllFaultSample()
{
    PhmClientInstance::getInstance()->InhibitAllFault();
}

void RecoverAllFaultSample()
{
    PhmClientInstance::getInstance()->RecoverInhibitAllFault();
}

int main(int argc, char* argv[])
{
	signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    hozon::netaos::log::InitLogging(
        "phm_client_sample",
        "phm_client_sample",
        hozon::netaos::log::LogLevel::kDebug,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "./",
        10,
        100
    );

    std::cout << "phm client sample start." << std::endl;
    // init
    PhmClientInstance::getInstance()->Init();

    ReportFaultSample();
    AliveTaskSample();
    DeadLineTaskSample();
    LogicTaskSample();
    InhibitFaultSample();
    RecoverFaultSample();
    InhibitAllFaultSample();
    RecoverAllFaultSample();
    ReportFaultRecoverySample();

    while(!stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // deinit
    PhmClientInstance::getInstance()->DeInit();
    std::cout << "phm client sample end." << std::endl;
	return 0;
}

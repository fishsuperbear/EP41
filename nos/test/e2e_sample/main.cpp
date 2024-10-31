#include <iostream>
#include "e2e/e2exf_cpp/include/e2exf_impl.h"

/*test*/
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <string>
#include <thread>
#include <random>
#include <chrono>
/*----*/

using namespace hozon::netaos::e2e;
std::mt19937 gen(std::random_device{}());
std::uniform_int_distribution<uint8_t> dist(0,255);
uint8_t DataIDList[16] = {174,124,106,58,47,154,237,220,98,37,173,212,59,125,101,188};
int main()
{
    E2E_SMConfigType SMConfig;
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

    E2EXf_ConfigType Config;
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

    uint32_t loop = 20;
    while(loop--) {
        for (int i = 0; i < dataLength; i++) data[i] = dist(gen);
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            ProtectResult Result = E2EXf_Protect(index, data, data.size());
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

            if(Result != ProtectResult::E_OK) {
                std::cout << "e2e protect error!\n";
                /*code*/
            }
            std::cout << "E2EXf_Protect Result: " << static_cast<int>(Result) 
                      << " Counter: " << std::setw(2) << (int)E2EXf_Mapping::Instance()->GetProtectState(index).P22ProtectState.Counter << std::string(16,' ')
                      << " duration: " << duration.count() << "ns" << std::endl;
        }

        {
            /*error make*/
            //data[7] = 0;
        }

        {
            auto start_time = std::chrono::high_resolution_clock::now();
            CheckResult Result = E2EXf_Check(index, data, data.size());
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            if(Result.GetProfileCheckStatus() != E2EXf_PCheckStatusType::E2E_P_OK) {
                std::cout << "e2e check error!\n";
                /*code*/
            }

            switch (Result.GetSMState())
            {
            case E2EXf_SMStateType::E2E_SM_VALID:
                std::cout << "e2e SMcheck success, message can use\n";
                break;

            case E2EXf_SMStateType::E2E_SM_INVALID:
                std::cout << "e2e SMcheck failed, message can't use\n";
                break;

            case E2EXf_SMStateType::E2E_SM_INIT:
                std::cout << "e2e SM is initializing, message can't use\n";
                /*code*/
                break;

            case E2EXf_SMStateType::E2E_SM_NODATA:
                std::cout << "e2e SM is waiting for the first data\n";
                /*code*/
                break;

            case E2EXf_SMStateType::E2E_SM_DEINIT:
                std::cout << "e2e SM is uninitialized\n";
                /*code*/
                break;

            default:
                break;
            }    

            std::cout << "E2EXf_Check   Result: " << Result.GetProfileCheckStatus()
                      << " SMState: " << Result.GetSMState() 
                      << " Counter: " << std::setw(2) << (int)E2EXf_Mapping::Instance()->GetCheckState(index).P22CheckState.Counter << std::string(5,' ')
                      << " duration: " << duration.count() << "ns" << std::endl << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    return 0;
}
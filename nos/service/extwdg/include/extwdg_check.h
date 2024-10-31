/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg
*/
#ifndef EXTWDG_CHECK_H_
#define EXTWDG_CHECK_H_



#include <cstdint>
#include <thread>
#include <csignal>
#include <mutex>
#include "extwdg_logger.h"
#include "phm_client_instance.h"

namespace hozon {
namespace netaos {
namespace extwdg {

#define CYCLE_PERIOD 5
#define SIGNAL38    38

class SetFlag
{
public:
    static SetFlag* getInstance();
    void SetReportFlag();
    void SetRecoverFlag();
    void SetCheckCase(std::string cases);
    std::string GetCheckCase();
    uint32_t GetReportFlag();
    uint32_t GetRecoverFlag();
    void ResetFlagZero();

private:
    SetFlag() {}

private:
    static SetFlag* instance_;
    static std::mutex mtx_;

private:
    uint32_t report_times_ = 0;
    uint32_t recover_times_ = 0;
    std::string cases_;
};

class SelfCheck
{
public:
    SelfCheck();
    ~SelfCheck() {}
    int32_t Init();
    void DeInit();
    int32_t RequestSelfCheck();
    int32_t RequestSoc2Mcu();

private:
    static int32_t AliveCheck();
    static int32_t DeadLineCheck();
    static int32_t LogicCheck();
    static void signalHandler(int signal, siginfo_t* pSigInfo, void* ucontext);

private:
    static bool soc2mcu_;

};

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif // EXTWDG_CHECK_H_
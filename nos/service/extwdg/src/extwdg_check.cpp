/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */
#include <future>
#include <iostream>
#include "extwdg_check.h"

namespace hozon {
namespace netaos {
namespace extwdg {

bool SelfCheck::soc2mcu_ = false;
SetFlag* SetFlag::instance_ = nullptr;
std::mutex SetFlag::mtx_;

SelfCheck::SelfCheck()
{

}

int32_t
SelfCheck::Init()
{
    struct sigaction sa;
    sa.sa_sigaction = &SelfCheck::signalHandler;
    sa.sa_flags = SA_SIGINFO;
    int res = sigaction(SIGNAL38, &sa, nullptr);
    if(res < 0) {
        EW_ERROR << "Register SIGNAL38 failed!";
        return -1;
    }
    PhmClientInstance::getInstance()->Init();
    return 0;
}

void
SelfCheck::DeInit()
{
    PhmClientInstance::getInstance()->DeInit();
}

int32_t
SelfCheck::RequestSelfCheck() 
{
    int32_t res, res1, res2, res3;
    std::future<int32_t> alivecheck = std::async(std::launch::async, AliveCheck);
    std::chrono::milliseconds timeout1(700);
    if(alivecheck.wait_for(timeout1) == std::future_status::timeout) {
        EW_ERROR << "AliveCheck Timeout";
        return -1;
    } 
    else {
            res1 = alivecheck.get();
            if(res1 != 0) {
                EW_ERROR << "AliveCheck failed";
            return -1;
        }
    }

    std::future<int32_t> deadlinecheck = std::async(std::launch::async, DeadLineCheck);
    std::chrono::milliseconds timeout2(500);
    if(deadlinecheck.wait_for(timeout2) == std::future_status::timeout) {
        EW_ERROR << "DeadLineCheck Timeout";
        return -1;
    }
    else {
            res2 = deadlinecheck.get();
            if(res2 != 0) {
                EW_ERROR << "DeadLineCheck failed";
            return -1;
        }
    }
    std::future<int32_t> logiccheck = std::async(std::launch::async, LogicCheck);
    std::chrono::milliseconds timeout3(500);
    if(logiccheck.wait_for(timeout3) == std::future_status::timeout) {
        EW_ERROR << "LogicCheck Timeout";
        return -1;
    }
    else {
            res3 = logiccheck.get();
            if(res3 != 0) {
                EW_ERROR << "LogicCheck failed";
            return -1;
        }
    }
    
    return 0;
}

void
SelfCheck::signalHandler(int signal, siginfo_t* pSigInfo, void* ucontext)
{
    EW_INFO << "SIGNAL38 is received!";
    if(nullptr != pSigInfo) {
        if(pSigInfo->si_int) {
            EW_INFO << "SIGNAL38's result is false!";
            soc2mcu_ = false;
        }
        else {
            EW_INFO << "SIGNAL38's result is true!";
            soc2mcu_ = true;
        }
    }
}

int32_t
SelfCheck::RequestSoc2Mcu()
{
    int32_t res = -1;
    if(soc2mcu_) {
        res = 0;
    }
    return res;
}

int32_t
SelfCheck::AliveCheck()
{
    EW_INFO << "AliveCheck enter!";
    int32_t res = -1;
    int checkPointId = 0;
    int periodTime = 100;
    std::string cases = "alive";

    PhmClientInstance::getInstance()->Start();
    SetFlag::getInstance()->SetCheckCase(cases);

    for(int i = 0; i < CYCLE_PERIOD; ++i) {
        if(i != 2) {
            EW_INFO << "count i is !"<< i;
            PhmClientInstance::getInstance()->ReportCheckPoint(checkPointId); 
        }
            std::this_thread::sleep_for(std::chrono::milliseconds(periodTime));
    }
    PhmClientInstance::getInstance()->Stop();
    EW_INFO << "report time = "<<SetFlag::getInstance()->GetReportFlag() << "recover_times = " <<SetFlag::getInstance()->GetRecoverFlag();
    if(SetFlag::getInstance()->GetReportFlag() == 1 && SetFlag::getInstance()->GetRecoverFlag() == 1) {
        EW_INFO << "AliveCheck success!";
        res = 0;
    }

    SetFlag::getInstance()->ResetFlagZero();

    return res;
}

int32_t
SelfCheck::DeadLineCheck()
{
    EW_INFO << "DeadLineCheck enter!";
    int32_t res = -1;
    std::vector<int> checkPointId = {1, 2};

    int periodTime1 = 30;
    int periodTime2 = 70;
    int periodTime3 = 120;

    std::string cases = "deadline";
    PhmClientInstance::getInstance()->Start();
    SetFlag::getInstance()->SetCheckCase(cases);

    for (auto& pointId : checkPointId) {
        PhmClientInstance::getInstance()->ReportCheckPoint(pointId);
        std::this_thread::sleep_for(std::chrono::milliseconds(periodTime1));
    }

    for (auto& pointId : checkPointId) {
        PhmClientInstance::getInstance()->ReportCheckPoint(pointId);
        std::this_thread::sleep_for(std::chrono::milliseconds(periodTime2));
    }

    for (auto& pointId : checkPointId) {
        PhmClientInstance::getInstance()->ReportCheckPoint(pointId);
        std::this_thread::sleep_for(std::chrono::milliseconds(periodTime3));
    }
    PhmClientInstance::getInstance()->Stop();
    EW_INFO << "report time = "<<SetFlag::getInstance()->GetReportFlag() << "recover_times = " <<SetFlag::getInstance()->GetRecoverFlag();
    if( SetFlag::getInstance()->GetReportFlag() == 2 && SetFlag::getInstance()->GetRecoverFlag() == 1) {
        EW_INFO << "DeadLineCheck success!";
        res = 0;
    }

    SetFlag::getInstance()->ResetFlagZero();

    return res;
}

int32_t
SelfCheck::LogicCheck()
{
    EW_INFO << "LogicCheck enter!";
    int32_t res = -1;
    std::vector<int> checkPointId1 = {7, 8, 7};
    std::vector<int> checkPointId2 = {7, 8, 10};

    std::string cases = "logic";
    PhmClientInstance::getInstance()->Start();
    SetFlag::getInstance()->SetCheckCase(cases);
    for (auto& pointId : checkPointId1) {
        PhmClientInstance::getInstance()->ReportCheckPoint(pointId);
    }

    for (auto& pointId : checkPointId2) {
        PhmClientInstance::getInstance()->ReportCheckPoint(pointId);
    }
    PhmClientInstance::getInstance()->Stop();
    EW_INFO << "report time = "<<SetFlag::getInstance()->GetReportFlag() << "recover_times = " <<SetFlag::getInstance()->GetRecoverFlag();
    if( SetFlag::getInstance()->GetReportFlag() == 1 && SetFlag::getInstance()->GetRecoverFlag() == 1) {
        EW_INFO << "LogicCheck success!";
        res = 0;
    }

    SetFlag::getInstance()->ResetFlagZero();

    return res;
}

SetFlag*
SetFlag::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new SetFlag();
        }
    }

    return instance_;
}


void
SetFlag::SetReportFlag()
{
    report_times_++;
}

void
SetFlag::SetRecoverFlag()
{
    recover_times_++;
}

void
SetFlag::SetCheckCase(std::string cases)
{
    cases_ = cases;
}

std::string
SetFlag::GetCheckCase()
{
    return cases_;
}

uint32_t
SetFlag::GetReportFlag()
{
    return report_times_;
}

uint32_t
SetFlag::GetRecoverFlag()
{
    return recover_times_;
}

void
SetFlag::ResetFlagZero()
{
    report_times_ = 0;
    recover_times_ = 0;
}

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon
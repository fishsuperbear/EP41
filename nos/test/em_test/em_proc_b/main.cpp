#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "em/include/emlogger.h"

#define RUN_DELAY_TIME "REPORT_RUN_DELAY_TIME"
#define TER_DELAY_TIME "REPORT_TER_DELAY_TIME"

using namespace hozon::netaos::em;

sig_atomic_t g_stopFlag = 0;

void HandlerSignal(int32_t sig)
{
    std::cout << "proc b sig:<<"<< sig << std::endl;
    g_stopFlag = 1;
}

void ActThread()
{
    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::seconds(1u));
    }
}

void InitLog()
{
    EMLogger::GetInstance().InitLogging("prob","em proc b",
        EMLogger::LogLevelType::LOG_LEVEL_TRACE,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "/log/", 10, 20
    );
    EMLogger::GetInstance().CreateLogger("-B");
}

int main(int argc, char* argv[])
{
    signal(SIGTERM, HandlerSignal);
    InitLog();

    char* run_delay_time = getenv(RUN_DELAY_TIME);
    if(run_delay_time){
        if( atoi(run_delay_time) > 0){ std::this_thread::sleep_for(std::chrono::seconds(atoi(run_delay_time))); }
    }

    std::shared_ptr<ExecClient> execli(new ExecClient());
    int32_t ret = execli->ReportState(ExecutionState::kRunning);
    if(ret){ std::cout << "b report fail." << std::endl; }

    for(int i=0; i<5; i++){
        _LOG_INFO<<" em_proc_b loop >>> "<<i;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::thread act(ActThread);
    act.join();

    char* ter_delay_time = getenv(TER_DELAY_TIME);
    if(ter_delay_time){
        if( atoi(ter_delay_time) > 0){ std::this_thread::sleep_for(std::chrono::seconds(atoi(ter_delay_time))); }
    }

    ret = execli->ReportState(ExecutionState::kTerminating);
    return 0;
}
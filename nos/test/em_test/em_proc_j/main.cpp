#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <signal.h>
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "em/include/emlogger.h"

using namespace hozon::netaos::em;

sig_atomic_t g_stopFlag = 0;

void HandlerSignal(int32_t sig)
{
    std::cout << "pro j sig:<<"<< sig << std::endl; 
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
    EMLogger::GetInstance().InitLogging("proj","em proc j",
        EMLogger::LogLevelType::LOG_LEVEL_TRACE,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "/log/", 10, 20
    );
    EMLogger::GetInstance().CreateLogger("-J");
}

int main(int argc, char* argv[])
{
    signal(SIGTERM, HandlerSignal); 
    InitLog();

    std::shared_ptr<ExecClient> execli(new ExecClient());
    int32_t ret = execli->ReportState(ExecutionState::kRunning);
    if(ret){ std::cout << "j report fail." << std::endl; }

    std::thread act(ActThread);
    act.join();

    ret = execli->ReportState(ExecutionState::kTerminating);
    return 0;
}
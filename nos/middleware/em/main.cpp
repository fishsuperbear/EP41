#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <signal.h>
#include "em/include/logger.h"
#include "em/include/execmanagement.h"
#include "sm/include/state_server.h"
#include "em/include/exec_server.h"

using namespace std;
using namespace hozon::netaos::em;
using namespace hozon::netaos::sm;

sig_atomic_t g_stopFlag = 0;

void HandlerSignal(int32_t sig)
{
    g_stopFlag = 1;
}

void ActThread()
{
    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::seconds(1u));
    }
}

// void InitLog()
// {
//     EManagerLogger::GetInstance().InitLogging("em","execution management",
//         EManagerLogger::LogLevelType::LOG_LEVEL_INFO, //the log level of application
//         hozon::netaos::log::HZ_LOG2FILE, //the output log mode
//         "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
//         10, //the max number log file , active when output log to file
//         20 //the max size of each  log file , active when output log to file
//     );
//     EManagerLogger::GetInstance().CreateLogger("em");
// }


int main(int argc, char* argv[])
{
    signal(SIGTERM, HandlerSignal);

    std::shared_ptr<ExecManagement> exemagr = ExecManagement::Instance();
    int32_t res = exemagr->Init();
    if(res != 0){
        LOG_ERROR<<"exec manager init failed";
        return 0;
    }

    std::shared_ptr<StateServer> statsvr = std::make_shared<StateServer>();
    statsvr->Start();

    std::shared_ptr<ExecServer> execsvr = std::make_shared<ExecServer>();
    execsvr->Start();

    /* start default mode */
    int32_t ret = 0;
    ret = exemagr->StartMode(exemagr->GetCurrMode());
    LOG_INFO<<"StartMode:"<< exemagr->GetCurrMode()<<" finished, ret:" <<ret;

    std::thread act(ActThread);
    act.join();

    exemagr->DeInit();
    execsvr->Stop();
    statsvr->Stop();

    return 0;
}

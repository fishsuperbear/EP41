#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <signal.h>
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "sys_statemgr/include/logger.h"
#include "sys_statemgr/include/sys_manager.h"
#include "sys_statemgr/include/phm_client_instance.h"

using namespace std;
using namespace hozon::netaos::em;
using namespace hozon::netaos::ssm;

sig_atomic_t g_stopFlag = 0;
int g_signum = 0;


void HandlerSignal(int32_t sig)
{
    g_signum = sig;
    g_stopFlag = 1;
}


int main(int argc, char* argv[])
{
    signal(SIGTERM, HandlerSignal);
    signal(SIGINT, HandlerSignal);
    SSM_LOG_INFO <<"ssm init";

    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    std::shared_ptr<SysManager> sysmgr = std::make_shared<SysManager>();
    execli->ReportState(ExecutionState::kRunning);

    if (sysmgr->Init() != 0) {
        SSM_LOG_ERROR<<"ssm init failed";
        uint32_t faultId = 4300;
        uint8_t faultObj = 1;
        uint8_t faultStatus = 1;
        SendFault_t sendOnceFault(faultId, faultObj, faultStatus);
        PhmClientInstance::getInstance()->ReportFault(sendOnceFault);
        return 0;
    }
    sysmgr->Run();

    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::microseconds(100u));
    }

    execli->ReportState(ExecutionState::kTerminating);
    sysmgr->DeInit();

    return 0;
}

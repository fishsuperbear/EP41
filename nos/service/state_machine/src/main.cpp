#include <unistd.h>
#include "config_param.h"
#include "state_machine.h"


int32_t main(int argc, char* argv[])
{
    // ara::exec::ExecutionClient execClient;
    // execClient.ReportExecutionState(ara::exec::ExecutionState::kRunning);
    if (argc != 2) {
        return 0;
    }

    // hozon::netaos::cfg::ConfigParam::Instance()->Init();

    StateMachine sm;
    sm.RegistAlgProcessFunc("statemachine", std::bind(&StateMachine::AlgProcess1, &sm, std::placeholders::_1));
    sm.Start(argv[1]);

    while (!sm.NeedStop()) {
        usleep(1000 * 1000);
    }

    sm.Stop();
    // hozon::netaos::cfg::ConfigParam::Instance()->DeInit();
    // execClient.ReportExecutionState(ara::exec::ExecutionState::kTerminating);
    return 0;
}


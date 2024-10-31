#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>

#include "data_identifier.h"
#include "routine_control.h"
#include "log/include/logging.h"
#include "diag/diag_agent/include/handler/diag_agent_handler.h"

bool stopFlag = false;

void SigHandler(int signum)
{
    std::cout << "--- diag sample sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = true;
}

int main(int argc, char* argv[])
{
	signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    hozon::netaos::log::InitLogging(
        "diag_agent_sample",
        "diag_agent_sample",
        hozon::netaos::log::LogLevel::kDebug,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "./",
        10,
        100
    );

    std::cout << "diag agent sample start." << std::endl;
    std::shared_ptr<DiagAgentDataIdentifier> dataIdentifier = std::make_shared<DataIdentifier>();
    std::shared_ptr<DiagAgentRoutineControl> routineControl = std::make_shared<RoutineControl>();
    DiagAgentHandler* handler = new DiagAgentHandler();
    // init
    int32_t initResult = handler->Init("/app/sample/diag_agent_sample/conf/diag_agent_config.json", dataIdentifier, routineControl);
    if (0 != initResult) {
        std::cout << "diag agent sample handler init failed failedcode: " << initResult << std::endl;
        if (nullptr != handler) {
            handler->DeInit();
            delete handler;
            handler = nullptr;
        }

        return initResult;
    }

    while(!stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // deinit
    if (nullptr != handler) {
        handler->DeInit();
        delete handler;
        handler = nullptr;
    }

    std::cout << "diag agent sample end." << std::endl;
	return 0;
}
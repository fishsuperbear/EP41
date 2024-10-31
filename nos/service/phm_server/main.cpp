#include <signal.h>
#include <thread>
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "phm_server/include/phm_server.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/common/phm_server_logger.h"

using namespace hozon::netaos::phm_server;
using namespace hozon::netaos::em;

PhmServer server;
void SigHandler(int signum)
{
    server.Stop();
}

void InitLog()
{
    PHMServerConfig::getInstance()->LoadPhmConfig();
    const PhmConfigInfo& configInfo = PHMServerConfig::getInstance()->GetPhmConfigInfo();
    PHMServerLogger::GetInstance().setLogLevel(static_cast<int32_t>(configInfo.LogLevel));
    uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE;
    if (0 == configInfo.LogMode) {
        outputMode = hozon::netaos::log::HZ_LOG2CONSOLE;
    }
    else if (2 == configInfo.LogMode) {
        outputMode = hozon::netaos::log::HZ_LOG2FILE | hozon::netaos::log::HZ_LOG2CONSOLE;
    }
    else {
        outputMode = hozon::netaos::log::HZ_LOG2FILE;
    }

    PHMServerLogger::GetInstance().InitLogging(configInfo.LogAppName,
        configInfo.LogAppDescription,
        static_cast<PHMServerLogger::PHMLogLevelType>(configInfo.LogLevel),
        outputMode,
        configInfo.LogFilePath,
        configInfo.MaxLogFileNum,
        configInfo.MaxSizeOfLogFile
    );

    PHMServerLogger::GetInstance().CreateLogger(configInfo.LogContextName);
}


void ActThread()
{
    pthread_setname_np(pthread_self(), "phm_server_run");
    server.Init();
    server.Run();
}

int main(int argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    InitLog();
    std::shared_ptr<ExecClient> spExecClient = std::make_shared<ExecClient>();
    spExecClient->ReportState(ExecutionState::kRunning);

    std::thread act(ActThread);
    act.join();
    spExecClient->ReportState(ExecutionState::kTerminating);
    return 0;
}

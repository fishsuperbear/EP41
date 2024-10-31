#include <signal.h>
#include <thread>
#include "diag/diag_server/include/diag_server.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"

using namespace hozon::netaos::diag;
using namespace hozon::netaos::em;

DiagServer server;

void SigHandler(int signum)
{
    server.Stop();
}

void InitLog()
{
    DiagServerConfig::getInstance()->LoadDiagConfig();
    DiagConfigInfo configInfo = DiagServerConfig::getInstance()->GetDiagConfigInfo();
    DiagServerLogger::GetInstance().setLogLevel(static_cast<int32_t>(configInfo.LogLevel));
    uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE;
    if (0 == configInfo.LogMode) {
        outputMode = hozon::netaos::log::HZ_LOG2CONSOLE;
    }
    else if (2 == configInfo.LogMode) {
        outputMode = hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE;
    }
    else {
        outputMode = hozon::netaos::log::HZ_LOG2FILE;
    }

    DiagServerLogger::GetInstance().InitLogging(configInfo.LogAppName,
        configInfo.LogAppDescription,
        static_cast<DiagServerLogger::DiagLogLevelType>(configInfo.LogLevel),
        outputMode,
        configInfo.LogFilePath,
        configInfo.MaxLogFileNum,
        configInfo.MaxSizeOfLogFile
    );

    DiagServerLogger::GetInstance().CreateLogger(configInfo.LogContextName);
}


void ActThread()
{
    pthread_setname_np(pthread_self(), "diag_server_run");
    server.Init();
    server.Run();
}

int main(int argc, char* argv[])
{
	signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    InitLog();
    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    execli->ReportState(ExecutionState::kRunning);

    std::thread act(ActThread);
    act.join();
    execli->ReportState(ExecutionState::kTerminating);
	return 0;
}

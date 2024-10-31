#include <signal.h>
#include <thread>

#include "remote_diag/include/remote_diag.h"
#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/common/remote_diag_logger.h"

using namespace hozon::netaos::remote_diag;

RemoteDiag server;
void SigHandler(int signum)
{
    server.Stop();
}

void InitLog()
{
    RemoteDiagConfig::getInstance()->LoadRemoteDiagConfig();
    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    RemoteDiagLogger::GetInstance().setLogLevel(static_cast<int32_t>(configInfo.LogLevel));
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


    RemoteDiagLogger::GetInstance().InitLogging(configInfo.LogAppName,
        configInfo.LogAppDescription,
        static_cast<RemoteDiagLogger::RemoteDiagLogLevelType>(configInfo.LogLevel),
        outputMode,
        configInfo.LogFilePath,
        configInfo.MaxLogFileNum,
        configInfo.MaxSizeOfLogFile
    );

    RemoteDiagLogger::GetInstance().CreateLogger(configInfo.LogContextName);
}


void ActThread()
{
    pthread_setname_np(pthread_self(), "remote_diag_run");
    server.Init();
    server.Run();
}

int main(int argc, char* argv[])
{
	signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    InitLog();

    std::thread act(ActThread);
    act.join();
    return 0;
}
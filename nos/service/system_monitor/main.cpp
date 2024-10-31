#include <signal.h>
#include <thread>

#include "system_monitor/include/system_monitor.h"
#include "system_monitor/include/common/system_monitor_config.h"
#include "system_monitor/include/common/system_monitor_logger.h"

using namespace hozon::netaos::system_monitor;

SystemMonitor server;
void SigHandler(int signum)
{
    server.Stop();
}

void InitLog()
{
    SystemMonitorConfig::getInstance()->LoadSystemMonitorConfig();
    const SystemMonitorConfigInfo& configInfo = SystemMonitorConfig::getInstance()->GetSystemMonitorConfigInfo();
    SystemMonitorLogger::GetInstance().setLogLevel(static_cast<int32_t>(configInfo.LogLevel));
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


    SystemMonitorLogger::GetInstance().InitLogging(configInfo.LogAppName,
        configInfo.LogAppDescription,
        static_cast<SystemMonitorLogger::SystemMonitorLogLevelType>(configInfo.LogLevel),
        outputMode,
        configInfo.LogFilePath,
        configInfo.MaxLogFileNum,
        configInfo.MaxSizeOfLogFile
    );

    SystemMonitorLogger::GetInstance().CreateLogger(configInfo.LogContextName);
}


void ActThread()
{
    pthread_setname_np(pthread_self(), "system_monitor_run");
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
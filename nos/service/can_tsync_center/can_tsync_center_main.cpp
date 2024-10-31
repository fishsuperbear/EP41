#include "can_tsync_center/can_tsync.h"
#include "can_tsync_center/can_tsync_logger.h"
#include "can_tsync_center/sig_stop.h"
#include <signal.h>

using namespace hozon::netaos::tsync;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Invalid argument number.\n";
        return -1;
    }

    hozon::netaos::SigHandler::Init();

    std::string config_file(argv[1]);
    LogConfig log_config;
    if (ConfigParser::ParseLogConfig(config_file, log_config) < 0) {
        std::cout << "Use default logger.\n";
        hozon::netaos::log::InitLogging(
                "CTSC",
                "Time sync over CAN",
                hozon::netaos::log::LogLevel::kInfo,
                hozon::netaos::log::HZ_LOG2FILE,
                "/opt/usr/log/soc_log/",
                10,
                10 * 1024 * 1024);
        CTSC_LOG_INFO << "Succ to init default logger.";
    }
    else {
        hozon::netaos::log::InitLogging(
                "CTSC",
                "Time sync over CAN",
                hozon::netaos::log::LogLevel(log_config.level),
                log_config.mode,
                log_config.file,
                10,
                10 * 1024 * 1024);
        CTSC_LOG_INFO << "Succ to init custom logger.";
    }


    std::vector<CanTSyncConfig> tsync_configs;
    if (ConfigParser::ParseCanTsyncConfig(config_file, tsync_configs) < 0) {
        CTSC_LOG_ERROR << "Invalid config file " << config_file;
        return -1;
    }

    std::vector<CanTsync> tsync_instances;
    tsync_instances.resize(tsync_configs.size());
    for (std::size_t i = 0; i < tsync_instances.size(); ++i) {
        CTSC_LOG_INFO << "Create tsync instance of " << tsync_configs[i].interface;
        int32_t ret = tsync_instances[i].Start(tsync_configs[i]);
        if (ret < 0) {
            CTSC_LOG_ERROR << "Fail to launch " << tsync_configs[i].interface;
        }
    }

    hozon::netaos::SigHandler::NeedStopBlocking();

    for (std::size_t i = 0; i < tsync_instances.size(); ++i) {
        CTSC_LOG_INFO << "Stop tsync instance of " << tsync_configs[i].interface;
        tsync_instances[i].Stop();
        CTSC_LOG_INFO << "Stop tsync instance of " << tsync_configs[i].interface << " end.";
    }

    return 0;
}
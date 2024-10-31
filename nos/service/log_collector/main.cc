#include <signal.h>

#include <gflags/gflags.h>

#include "logblock_helper/include/log_block_batch_consumer.h"

#include "log_collector/include/log_file_manager.h"
#include "log_collector/include/process_data_handler.h"
#include "log_collector/include/utils.h"
#include "log_collector/include/config_manager.h"
#include "log_collector/include/log_collector_logger.h"

using hozon::netaos::logblock::LogBlockBatchConsumer;
using hozon::netaos::logcollector::LogFileManager;
using hozon::netaos::logcollector::ProcessDataHandler;
using hozon::netaos::logcollector::CommonTool;
using hozon::netaos::logcollector::ConfigManager;
using hozon::netaos::logcollector::LogCollectorLogger;

//DEFINE_string(config_file, "/app/conf/log_collector_config.json", "config file");
DEFINE_string(config_file, "/map/zhaoxin/xumengjun/nos_orin/conf/log_collector_config.json", "config file");

void SignalHandler(int sig) {
    LogBlockBatchConsumer::Instance().Stop();
    ProcessDataHandler::Instance().Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

bool InitLog() {
    auto log_level = ConfigManager::Instance().LogLevel();
    LogCollectorLogger::GetInstance().InitLogging(ConfigManager::Instance().LogAppName(),
                ConfigManager::Instance().LogAppDesc(),
                static_cast<LogCollectorLogger::LogCollectorLogLevelType>(log_level),
                ConfigManager::Instance().LogMode(),
                ConfigManager::Instance().LogFilePath(),
                ConfigManager::Instance().MaxLogFileNum(),
                ConfigManager::Instance().MaxLogFileSize()
                );

    auto &log_context_name = ConfigManager::Instance().LogContextName();
    LogCollectorLogger::GetInstance().CreateLogger(log_context_name);

    return true;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    signal(SIGTERM, SignalHandler);
    signal(SIGKILL, SignalHandler);
    signal(SIGINT, SignalHandler);

    if (!ConfigManager::Instance().LoadConfig(FLAGS_config_file)) {
        return -1;
    }

    if (!InitLog()) {
        return -1;
    }

    if (!LogFileManager::LoadGlobalLogConfig(ConfigManager::Instance().GlobalLogConfigFile())) {
        LOG_COLLECTOR_ERROR << "load global log config failed, config file:"
                            << ConfigManager::Instance().GlobalLogConfigFile();
        return -1;
    }
    LogFileManager::LoadHistoryLogFiles();

    ProcessDataHandler::Instance().Start();

    auto data_callback_func = [&](const char *appid, unsigned int process_id, unsigned int thread_id, 
                                    const iovec *iov, size_t count, size_t len) {
        LogFileManager::Instance().GetLogFileWriter(appid, process_id, len)->AddData(iov, count, len);
    };
    if (!LogBlockBatchConsumer::Instance().Start(ConfigManager::Instance().ConsumerThreadNum(),
                    data_callback_func)) {
        LOG_COLLECTOR_ERROR << "start log block consumer failed.";
        return -1;
    }

    LogBlockBatchConsumer::Instance().Wait();
    ProcessDataHandler::Instance().Wait();

    gflags::ShutDownCommandLineFlags();

    return 0;
}

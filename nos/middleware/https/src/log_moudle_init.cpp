/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: log_module_init.h is designed for https.
 */
#ifndef LOG_MODULE_INIT_H_
#define LOG_MODULE_INIT_H
// #include <memory>
// #include <atomic>
#include "log_moudle_init.h"
#include "https_logger.h"

namespace hozon {
namespace netaos {
namespace https {

std::shared_ptr<LogModuleInit> LogModuleInit::logmoule_init_(new LogModuleInit());
std::once_flag initFlag;
LogModuleInit::LogModuleInit () {

}

std::shared_ptr<LogModuleInit> LogModuleInit::getInstance(){
    return logmoule_init_;
}

void LogModuleInit::initLog() {
   std::call_once(initFlag, [](){
        // std::cout <<"LogModuleInit initLog, must be called only once !\n";
        // SecurehttpServerLogger::GetInstance().setLogLevel(static_cast<int32_t>(
        // SecurehttpServerLogger::SecurehttpLogLevelType::SECUREHTTP_DEBUG));
        // SecurehttpServerLogger::GetInstance().InitLogging(
        // "SECUREHTTP",       // the id of application
        // "SECUREHTTP test",  // the log id of application
        // SecurehttpServerLogger::SecurehttpLogLevelType::
        //     SECUREHTTP_DEBUG,  // the log
        //                         // level of
        //                         // application
        // hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
        //     "/opt/usr/log/soc_log/",  // the log file directory, active when output log to file
        //     10,    // the max number log file , active when output log to file
        //     20     // the max size of each  log file , active when output log to file
        // );
        HttpsLogger::GetInstance().CreateLogger("HTTPS");
    });
}

}  // namespace https
}  // namespace netaos
}  // namespace hozon
#endif
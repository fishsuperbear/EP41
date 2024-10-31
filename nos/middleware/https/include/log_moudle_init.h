/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: log_module_init.h is designed for Https.
 */
#ifndef LOG_MODULE_INIT_H_
#define LOG_MODULE_INIT_H_
#include <memory>
#include <mutex>
#include <atomic>
#include "https_logger.h"
namespace hozon {
namespace netaos {
namespace https {

class LogModuleInit {
public:

    ~LogModuleInit() = default;
    static std::shared_ptr<LogModuleInit> getInstance();

    void initLog();

private:
    LogModuleInit();
    LogModuleInit(const LogModuleInit &);  
    LogModuleInit & operator = (const LogModuleInit &);

private:
    static std::shared_ptr<LogModuleInit> logmoule_init_;
};

}  // namespace Https
}  // namespace netaos
}  // namespace hozon
#endif
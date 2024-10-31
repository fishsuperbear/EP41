#pragma once

#include <iostream>
#include <memory>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logger.h"
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace sensor {
class Sensorlogger {
private:
    /* data */
    Sensorlogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
public:
    static Sensorlogger& GetInstance() {
        static Sensorlogger instance;
        return instance;
    }
    ~Sensorlogger() { }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const {return logger_; }

    void InitLogger(
        // std::string log_name,
        // std::string log_desciription,
        log::LogLevel log_level,
        uint32_t log_mode,
        std::string log_path) {
            hozon::netaos::log::InitLogging(
                "SSTR",
                "Sensor Trans",
                log_level,
                log_mode,
                log_path,
                10,
                20);
    }

    void CreateLogger(log::LogLevel log_level) {
        auto log_{hozon::netaos::log::CreateLogger("SSTR", "Sensor Trans", log_level)};
        logger_ = log_;
    }

};

#define SENSOR_LOG_HEAD   getpid() << " " << (long int)syscall(__NR_gettid) << " " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "

#define SENSOR_LOG_CRITICAL             (Sensorlogger::GetInstance().GetLogger()->LogCrirical() << SENSOR_LOG_HEAD)
#define SENSOR_LOG_ERROR                (Sensorlogger::GetInstance().GetLogger()->LogError() << SENSOR_LOG_HEAD)
#define SENSOR_LOG_WARN                 (Sensorlogger::GetInstance().GetLogger()->LogWarn() << SENSOR_LOG_HEAD)
#define SENSOR_LOG_INFO                 (Sensorlogger::GetInstance().GetLogger()->LogInfo() << SENSOR_LOG_HEAD)
#define SENSOR_LOG_DEBUG                (Sensorlogger::GetInstance().GetLogger()->LogDebug() << SENSOR_LOG_HEAD)
#define SENSOR_LOG_TRACE                (Sensorlogger::GetInstance().GetLogger()->LogTrace() << SENSOR_LOG_HEAD)


class SENSOREarlyLogger {
public:
    ~SENSOREarlyLogger(){
        std::cout << std::endl;
    }

    template<typename T>
    SENSOREarlyLogger& operator<<(const T& value) {
        std::cout << value;
        return *this;
    }
};

#define SENSOR_EARLY_LOG     SENSOREarlyLogger() << SENSOR_LOG_HEAD

}   // namespace sensor
}   // namespace netaos
}   // namespace hozon
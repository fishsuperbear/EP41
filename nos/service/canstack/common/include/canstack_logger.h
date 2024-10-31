#pragma once
#include <memory>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"


using namespace hozon::netaos::log;

namespace hozon {
namespace netaos {
namespace canstack {
class SensorLogger {
public:
    static SensorLogger& GetInstance() {
        static SensorLogger instance;
        return instance;
    }
    ~SensorLogger() {}

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const {return logger_; }

    void InitLogger(
        std::string log_name,
        // std::string log_description,
        LogLevel log_level,
        uint32_t log_mode,
        std::string log_path) {
            hozon::netaos::log::InitLogging(
                log_name,
                "Canstack Loger",
                log_level,
                log_mode,
                log_path,
                10,
                20);
        }
    void CreateLogger(std::string ctx_id, std::string ctx_description, LogLevel log_level) {
        auto log_{hozon::netaos::log::CreateLogger(ctx_id, ctx_description, log_level)};
        logger_ = log_;
    }

private:
    SensorLogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_;

};
#define SENSOR_LOG_HAED getpid() << " " << (long int)syscall(__NR_gettid) \
    << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << "(" << __LINE__ << ") | "


#define CAN_LOG_CRITICIAL  hozon::netaos::canstack::SensorLogger::GetInstance().GetLogger()->LogCritical() << SENSOR_LOG_HAED
#define CAN_LOG_ERROR      hozon::netaos::canstack::SensorLogger::GetInstance().GetLogger()->LogError() << SENSOR_LOG_HAED
#define CAN_LOG_WARN       hozon::netaos::canstack::SensorLogger::GetInstance().GetLogger()->LogWarn() << SENSOR_LOG_HAED
#define CAN_LOG_INFO       hozon::netaos::canstack::SensorLogger::GetInstance().GetLogger()->LogInfo() << SENSOR_LOG_HAED
#define CAN_LOG_DEBUG      hozon::netaos::canstack::SensorLogger::GetInstance().GetLogger()->LogDebug() << SENSOR_LOG_HAED
#define CAN_LOG_TRACE      hozon::netaos::canstack::SensorLogger::GetInstance().GetLogger()->LogTrace() << SENSOR_LOG_HAED

class SensorEarlyLogger {
public: 
    ~SensorEarlyLogger() {
        std::cout << std::endl;
    }

    template<typename T> 
    SensorEarlyLogger& operator<<(const T& value) {
        std::cout << value;
        return *this;
    }
};

#define CAN_ERALY_LOG  SensorEarlyLogger() << SENSOR_LOG_HAED

}
}
} // namespace hozon



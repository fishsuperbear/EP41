#include "logging.h"

namespace hozon {
namespace netaos {

using namespace  hozon::netaos::log;

class HZMMLogger {
   public:
    static HZMMLogger& GetInstance() {
        static HZMMLogger instance;
        return instance;
    }

    ~HZMMLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    void InitLogging(std::string appId = "MM",                        // the log id of application
                     std::string appDescription = "Map Manage Module",       // the log id of application
                     LogLevel appLogLevel = LogLevel::kInfo,                  //the log level of application
                     std::uint32_t outputMode = HZ_LOG2FILE,                  //the output log mode
                     std::string directoryPath = "/opt/usr/log/soc_log/",                         //the log file directory, active when output log to file
                     std::uint32_t maxLogFileNum = 10,                         //the max number log file , active when output log to file
                     std::uint64_t maxSizeOfLogFile = 20                       //the max size of each  log file , active when output log to file
    ) {
        hozon::netaos::log::InitLogging(appId, appDescription, appLogLevel, outputMode, directoryPath, maxLogFileNum, maxSizeOfLogFile);
        logger_ = hozon::netaos::log::CreateLogger(appId, appDescription, appLogLevel);
    }

   private:
    HZMMLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};


#define HZ_MM_LOG_CRITICAL (HZMMLogger::GetInstance().GetLogger()->LogCritical())
#define HZ_MM_LOG_ERROR (HZMMLogger::GetInstance().GetLogger()->LogError())
#define HZ_MM_LOG_WARN (HZMMLogger::GetInstance().GetLogger()->LogWarn())
#define HZ_MM_LOG_INFO (HZMMLogger::GetInstance().GetLogger()->LogInfo())
#define HZ_MM_LOG_DEBUG (HZMMLogger::GetInstance().GetLogger()->LogDebug())
#define HZ_MM_LOG_TRACE (HZMMLogger::GetInstance().GetLogger()->LogTrace())
  // namespace
}  // namespace netaos
}  // namespace hozon


#include "logging.h"

class DeSayLogger
{
public:
    DeSayLogger() : logger_(nullptr) {};
    virtual ~DeSayLogger() {};

    enum class DSLogLevelType {
        DS_TRACE = 0,
        DS_DEBUG = 1,
        DS_INFO = 2,
        DS_WARN = 3,
        DS_ERROR = 4,
        DS_CRITICAL = 5,
        DS_OFF = 6
    };

    hozon::netaos::log::LogLevel DeSayParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<DSLogLevelType>(logLevel);
        switch (type) {
            case DSLogLevelType::DS_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DSLogLevelType::DS_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DSLogLevelType::DS_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DSLogLevelType::DS_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DSLogLevelType::DS_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DSLogLevelType::DS_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DSLogLevelType::DS_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    void InitLogger(std::string logCfgFile)
    {
        hozon::netaos::log::InitLogging(
            logCfgFile
        );
    }

    void CreateLogger(const std::string ctxId = "DESAY")
    {
        const hozon::netaos::log::LogLevel level = DeSayParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static DeSayLogger& GetInstance()
    {
        static DeSayLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> DSGetLogger() const
    {
        return logger_;
    }

    int32_t getLogLevel()
    {
        return level_;
    }

    void setLogLevel(int32_t level)
    {
        level_ = level;
    }

private:
    DeSayLogger(const DeSayLogger&);
    DeSayLogger& operator=(const DeSayLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(DSLogLevelType::DS_TRACE);
};

#define DF_LOG_CRITICAL             (DeSayLogger::GetInstance().DSGetLogger()->LogCritical() << hozon::netaos::log::FROM_HERE)
#define DF_LOG_ERROR                (DeSayLogger::GetInstance().DSGetLogger()->LogError() << hozon::netaos::log::FROM_HERE)
#define DF_LOG_WARN                 (DeSayLogger::GetInstance().DSGetLogger()->LogWarn() << hozon::netaos::log::FROM_HERE)
#define DF_LOG_INFO                 (DeSayLogger::GetInstance().DSGetLogger()->LogInfo() << hozon::netaos::log::FROM_HERE)
#define DF_LOG_DEBUG                (DeSayLogger::GetInstance().DSGetLogger()->LogDebug() << hozon::netaos::log::FROM_HERE)
#define DF_LOG_TRACE                (DeSayLogger::GetInstance().DSGetLogger()->LogTrace() << hozon::netaos::log::FROM_HERE)
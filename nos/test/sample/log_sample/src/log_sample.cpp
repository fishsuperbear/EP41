#include <memory>
#include <map>
#include "logging.h"

using namespace hozon::netaos::log;


class DefaultLogger {
public:
    static DefaultLogger& GetInstance() {
        static DefaultLogger instance;
        return instance;
    }
    ~DefaultLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }


    void InitLogging(std::string appId = "DEFAULT_APP",  // the log id of application
        std::string appDescription = "default application", // the log id of application
        LogLevel appLogLevel = LogLevel::kTrace, //the log level of application
        std::uint32_t outputMode = HZ_LOG2CONSOLE | HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "./", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    ){
        hozon::netaos::log::InitLogging(
            appId,
            appDescription,
            appLogLevel,
            outputMode,
            directoryPath,
            maxLogFileNum,
            maxSizeOfLogFile
        );

        logger_ = hozon::netaos::log::CreateLogger("default_ctx_id", "default logger instance", hozon::netaos::log::LogLevel::kTrace);
    }

    void InitLogger(std::string logCfgFile)
    {
        hozon::netaos::log::InitLogging(
            logCfgFile
        );

        logger_ = hozon::netaos::log::CreateLogger("default_ctx_id", "default logger instance", hozon::netaos::log::LogLevel::kTrace);
    }

private:
    DefaultLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

// 输出附带文件、行号等信息
#define DF_LOG_CRITICAL             (DefaultLogger::GetInstance().GetLogger()->LogCritical())
#define DF_LOG_ERROR                (DefaultLogger::GetInstance().GetLogger()->LogError())
#define DF_LOG_WARN                 (DefaultLogger::GetInstance().GetLogger()->LogWarn())
#define DF_LOG_INFO                 (DefaultLogger::GetInstance().GetLogger()->LogInfo())
#define DF_LOG_DEBUG                (DefaultLogger::GetInstance().GetLogger()->LogDebug())
#define DF_LOG_TRACE                (DefaultLogger::GetInstance().GetLogger()->LogTrace())

// 判断某个日志等级是否有效, 并且输出空字符的形式，不输出日志，从而节约资源
#define DF_LOG_ENABLE(level) DefaultLogger::GetInstance().GetLogger()->IsEnabled(level)
#define DF_OPTIMISE_LOG_CRITICAL (!DF_LOG_ENABLE(LogLevel::kCritical)) ? (DF_LOG_CRITICAL << "") : (DF_LOG_CRITICAL << FROM_HERE)
#define DF_OPTIMISE_LOG_ERROR    (!DF_LOG_ENABLE(LogLevel::kError))    ? (DF_LOG_ERROR << "")    : (DF_LOG_ERROR << FROM_HERE)
#define DF_OPTIMISE_LOG_WARN     (!DF_LOG_ENABLE(LogLevel::kWarn))     ? (DF_LOG_WARN << "")     : (DF_LOG_WARN << FROM_HERE)
#define DF_OPTIMISE_LOG_INFO     (!DF_LOG_ENABLE(LogLevel::kInfo))     ? (DF_LOG_INFO << "")     : (DF_LOG_INFO << FROM_HERE)
#define DF_OPTIMISE_LOG_DEBUG    (!DF_LOG_ENABLE(LogLevel::kDebug))    ? (DF_LOG_DEBUG << "")    : (DF_LOG_DEBUG << FROM_HERE)
#define DF_OPTIMISE_LOG_TRACE    (!DF_LOG_ENABLE(LogLevel::kTrace))    ? (DF_LOG_TRACE << "")    : (DF_LOG_TRACE << FROM_HERE)



int main(int argc, char* argv[]) {
    // Log初始化方式 1
    // DefaultLogger::GetInstance().InitLogging();

    // Log初始化方式 2
    DefaultLogger::GetInstance().InitLogger("../conf/log_cfg.json");

    bool bBoolValue = false;
    std::string bStringValue = "false";
    uint64_t bUint64Value = 12345678890;
    double dTestlValue = 1234.5678;
    float fTestlValue = 1234.5678;

    // 日志输出举例
    DF_LOG_INFO << "123";
    DF_LOG_INFO << "bBoolValue : " << bBoolValue;
    DF_LOG_INFO << "bStringValue : " << bStringValue;
    DF_LOG_INFO << "bUint64Value : " << bUint64Value;
    DF_LOG_INFO << "dTestlValue : " << dTestlValue;
    DF_LOG_INFO << "fTestlValue : " << fTestlValue;
    
    // 手动调整浮点类型输出精度并使用fixed
    DF_LOG_INFO << SET_PRECISION(1) << FIXED << "dTestlValue : " << dTestlValue;

    DF_OPTIMISE_LOG_TRACE << "befor SetLogLevel(kError)";

    // 动态设置日志等级
    DefaultLogger::GetInstance().GetLogger()->SetLogLevel(LogLevel::kError);

    DF_OPTIMISE_LOG_TRACE << "after SetLogLevel(kError)";

    // 获取关键日志举例
    auto operation_log_app = CreateOperationLogger("APP", "description", LogLevel::kTrace);

    // 手动调整浮点类型输出精度
    DF_LOG_INFO << SET_PRECISION(2) << "dTestlValue : " << dTestlValue;

    // 输出关键日志举例
    operation_log_app->LogCritical() << "Critical log of application, 123";
    operation_log_app->LogCritical() << "Critical log of application, bBoolValue : " << bBoolValue;
    operation_log_app->LogCritical() << "Critical log of application, bStringValue : " << bStringValue;
    operation_log_app->LogCritical() << "Critical log of application, bUint64Value : " << bUint64Value;
    operation_log_app->LogCritical() << "Critical log of application, dTestlValue : " << dTestlValue;
    operation_log_app->LogCritical() << "Critical log of application, fTestlValue : " << fTestlValue;

    return 0;
}


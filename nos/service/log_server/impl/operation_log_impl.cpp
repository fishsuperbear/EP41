#include <sys/syslog.h>

#include "log/include/global_log_config_manager.h"

#include "log_server/impl/operation_log_impl.h"
#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {

using HzGlobalLogConfigManager = hozon::netaos::log::HzGlobalLogConfigManager;

OperationLogImpl::OperationLogImpl()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
    trace_.clear();
}

int32_t
OperationLogImpl::Init()
{
    LOG_SERVER_INFO << "OperationLogImpl::Init";
    auto res = Start(operation_log_service_name);
    LOG_SERVER_INFO << "OperationLogImpl::Init Done";
    return res;
}

int32_t
OperationLogImpl::DeInit()
{
    LOG_SERVER_INFO << "OperationLogImpl::DeInit";
    auto res = Stop();
    LOG_SERVER_INFO << "OperationLogImpl::DeInit Done";
    return res;
}

int32_t
OperationLogImpl::Process(const std::string& request, std::string& reply)
{
    std::lock_guard<std::mutex> lck(mtx_);
    LOG_SERVER_DEBUG << "OperationLogImpl::Process";
    std::string data(request.begin(), request.end());

    LogoutInfo info{};
    info.ParseFromString(data);

    std::string appID = info.app_id();
    std::string ctxID = info.ctx_id();
    LogLevel level = convertToLogLevel(info.log_level());
    std::string message = info.message();

    LOG_SERVER_DEBUG << "ParseFromArray() appID is : " << appID;
    LOG_SERVER_DEBUG << "ParseFromArray() ctxID is : " << ctxID;

    LOG_SERVER_DEBUG << "ParseFromArray() logLevel is : " << static_cast<uint32_t>(level);
    LOG_SERVER_DEBUG << "ParseFromArray() message is : " << message;

    Logout(appID, ctxID, level, message);

    reply.clear();
    return 0;
}

void
OperationLogImpl::Logout(const std::string& appid, const std::string& ctxid, LogLevel level, const std::string& message)
{
    auto logOut = GetTraceByCtxID(ctxid, appid);
    auto str = "[" + appid + "] " + message;
    switch (level)
    {
    case LogLevel::kTrace:
        logOut->trace_log(str);
        break;
    case LogLevel::kDebug:
        logOut->debug_log(str);
        break;
    case LogLevel::kInfo:
        logOut->info_log(str);
        break;
    case LogLevel::kWarn:
        logOut->warn_log(str);
        break;
    case LogLevel::kError:
        logOut->error_log(str);
        break;
    case LogLevel::kCritical:
        logOut->critical_log(str);
        break;
    default:
        break;
    }
}

std::shared_ptr<hozon::netaos::log::HzOperationLogTrace>
OperationLogImpl::GetTraceByCtxID(const std::string& ctxid, const std::string& appid)
{
    auto it = trace_.find(ctxid);
    if (it != trace_.end())
    {
        LOG_SERVER_DEBUG << "this ctx be created, ctxid is : " << ctxid;
        return it->second;
    }
    else
    {
        LOG_SERVER_DEBUG << "creating file trace, ctxid is : " << ctxid;
        std::string optional_path = "";

        #ifdef BUILD_FOR_MDC
            optional_path = OPERATION_LOG_PATH_FOR_MDC_LLVM;
        #elif BUILD_FOR_J5
            optional_path = OPERATION_LOG_PATH_FOR_J5;
        #elif BUILD_FOR_ORIN
            optional_path = OPERATION_LOG_PATH_FOR_ORIN;
        #else
            optional_path = OPERATION_LOG_PATH_DEFAULT;
        #endif
        // 针对每个不同的ctxID，创建不同的文件
        auto trace_ptr = std::make_shared<hozon::netaos::log::HzOperationLogTrace>();
        std::string logFileName = ctxid;
        std::string filePath = optional_path + ctxid + ".log";

        std::uint32_t  maxLogFileStoredNum = 10;
        std::uint64_t  maxSizeOfEachLogFile = 1 * 1024 * 1024;

        std::uint32_t log_mode = 2;
        if (HzGlobalLogConfigManager::GetInstance().LoadConfig()) {
            const auto &log_config = HzGlobalLogConfigManager::GetInstance().GetAppLogConfig(ctxid);
            if (log_config && log_config->hasLogMode) {
                log_mode = log_config->logMode;
            }
            if (log_config && log_config->hasLogPath) {
                filePath = log_config->logPath + ctxid + ".log";
            }
            if (log_config && log_config->hasMaxLogFileNum) {
                maxLogFileStoredNum = log_config->maxLogFileNum;
            }
            if (log_config && log_config->hasMaxSizeOfEachLogFile) {
                maxSizeOfEachLogFile = log_config->maxSizeOfEachLogFile;
            }
        }

        if ((log_mode & hozon::netaos::log::HZ_LOG2FILE) > 0) {
            trace_ptr->setLog2File(logFileName, filePath, maxLogFileStoredNum, maxSizeOfEachLogFile);
        }

        if ((log_mode & hozon::netaos::log::HZ_LOG2LOGSERVICE) > 0) {
            trace_ptr->setLog2LogService(logFileName);
        }

        trace_ptr->initDevice();
        trace_.insert(std::make_pair(ctxid, trace_ptr));
        return trace_ptr;
    }
}

LogLevel
OperationLogImpl::convertToLogLevel(uint32_t value)
{
    switch (value)
    {
        case 0x00U:
            return LogLevel::kTrace;
        case 0x01U:
            return LogLevel::kDebug;
        case 0x02U:
            return LogLevel::kInfo;
        case 0x03U:
            return LogLevel::kWarn;
        case 0x04U:
            return LogLevel::kError;
        case 0x05U:
            return LogLevel::kCritical;
        case 0x06U:
            return LogLevel::kOff;
        default:
            // 默认值
            return LogLevel::kInfo;
    }
}


}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

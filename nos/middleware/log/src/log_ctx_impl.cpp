#include "log_ctx_impl.hpp"
#include "log_manager.hpp"

namespace hozon {
namespace netaos {
namespace log {

CtxImpl::CtxImpl(Logger &selfLogger, std::string  ctxLogId, std::string ctxLogDescription, LogLevel ctxDefLogLevel)
    :ctxLogId_(ctxLogId)
    ,ctxLogDescription_(ctxLogDescription)
    ,ctxDefLogLevel_(ctxDefLogLevel)
    ,forceOutput_(false)
   // ,finalOutputLogLevel_(HzLogManager::GetInstance()->getAppLogLevel() > ctxDefLogLevel ?  HzLogManager::GetInstance()->getAppLogLevel() : ctxDefLogLevel)
    ,finalOutputLogLevel_(ctxDefLogLevel)
    ,m_logger(selfLogger)
{
}

std::string  CtxImpl::getCtxLogId()
{
    return ctxLogId_;
}

std::string  CtxImpl::getCtxLogDescription()
{
    return ctxLogDescription_;
}

LogLevel  CtxImpl::getCtxLogLevel()
{
    return ctxDefLogLevel_;
}

LogLevel  CtxImpl::getOutputLogLevel()
{
    return finalOutputLogLevel_;
}

bool CtxImpl::IsEnabled(LogLevel level)
{
    if (finalOutputLogLevel_ <= level) {
        return true;
    }
    return false;
}

void CtxImpl::UpdateAppLogLevel(const LogLevel appLogLevel)
{
    if (!forceOutput_) {
        // normalSetCtxLogLevel(appLogLevel > ctxDefLogLevel_ ?  appLogLevel : ctxDefLogLevel_);
        normalSetCtxLogLevel(ctxDefLogLevel_);
    }
}

void CtxImpl::normalSetCtxLogLevel(const LogLevel level)
{
    if (!forceOutput_) {
        ctxDefLogLevel_ = level;
        // setOutputLogLevel(HzLogManager::GetInstance()->getAppLogLevel() > level ?  HzLogManager::GetInstance()->getAppLogLevel() : level);
        setOutputLogLevel(level);
    }
}

void CtxImpl::forceSetCtxLogLevel(const LogLevel level)
{
    forceOutput_ = true;
    setOutputLogLevel(level);
}

void CtxImpl::setOutputLogLevel(const LogLevel level)
{
    finalOutputLogLevel_ = level;
}

void CtxImpl::LogOut(LogLevel level, const std::string& message)
{
    if (level >= finalOutputLogLevel_) {
        HzLogManager::GetInstance()->logout(level, message);
    }
}

}
}
}
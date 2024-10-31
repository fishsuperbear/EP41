#include "util/log_util.h"

#include <stdarg.h>

#include <chrono>
#include <ctime>
#include <sstream>


#include <syslog.h>

#include "util/time_util.h"
#include "util/default_time_util.h"

namespace advc {


    std::string LogUtil::GetLogPrefix(int level) {
        std::stringstream ss;
        char buf [64];
        std::time_t now = Time::time_util->getLocalTime();
        std::strftime(buf, 64, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        ss << buf;
        ss << " " << LOG_LEVEL_STRING(level);
        return ss.str();
    }

    std::string LogUtil::FormatLog(int level, const char *fmt, ...) {
        std::stringstream ss;
        ss << GetLogPrefix(level);
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, 1024, fmt, ap);
        va_end(ap);
        ss << buf;
        return ss.str();
    }

    void LogUtil::Syslog(int level, const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        syslog(level, fmt, ap);
        va_end(ap);
    }

}  // namespace advc

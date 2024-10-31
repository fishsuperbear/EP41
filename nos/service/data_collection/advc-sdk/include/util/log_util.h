#ifndef LOG_UTIL_H
#define LOG_UTIL_H

#include <stdint.h>
#include <string>
#include "advc_defines.h"


namespace advc {

    typedef void (*LogCallback)(const std::string &logstr);

    class LogUtil {
    public:

        /**
         * @brief Get the Log Prefix object
         * 
         * @param level 
         * @return std::string 
         */
        static std::string GetLogPrefix(int level);
        
        /**
         * @brief format Log
         * 
         * @param level 
         * @param fmt 
         * @param ... 
         * @return std::string 
         */
        static std::string FormatLog(int level, const char *fmt, ...);

        /**
         * @brief 
         * 
         * @param level 
         * @param fmt 
         * @param ... 
         */
        static void Syslog(int level, const char *fmt, ...);
    };

}  // namespace advc

#endif

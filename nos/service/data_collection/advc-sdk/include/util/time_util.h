#ifndef TIME_UTIL_H
#define TIME_UTIL_H
#include <string>
#include <time.h>
#include<iostream>
#include<memory>
namespace advc {
    class Time {
    public:
        Time() {};
        ~Time() {};
        static std::shared_ptr<Time> time_util;

        /**
         * @brief Get the Local Time object
         * 
         * @return time_t 
         */
        virtual time_t getLocalTime()=0;

        /**
         * @brief Get the Unix Time object
         * 
         * @return time_t 
         */
        virtual time_t getUnixTime()=0;
        };
} // namespace advc

#endif
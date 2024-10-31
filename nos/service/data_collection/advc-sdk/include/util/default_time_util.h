#ifndef DEFAULT_TIME_UTIL_H
#define DEFAULT_TIME_UTIL_H
#include "time_util.h"
#include <string>
#include <time.h>
namespace advc {
   
    class  DefaultTimeUtil : public Time {
        public:            
             DefaultTimeUtil() {};
             ~DefaultTimeUtil() {};

            /**
             * @brief Get the Local Time object
             * 
             * @return time_t 
             */
             virtual  time_t getLocalTime();

             /**
              * @brief Get the Unix Time object
              * 
              * @return time_t 
              */
             virtual  time_t getUnixTime();
            };
} // namespace advc

#endif
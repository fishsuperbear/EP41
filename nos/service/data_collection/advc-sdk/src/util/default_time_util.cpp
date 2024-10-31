#include "util/default_time_util.h"

#include <iostream>
namespace advc {

time_t DefaultTimeUtil::getLocalTime() {
    time_t timep;
    time(&timep);
    return timep;
}

time_t DefaultTimeUtil::getUnixTime() {
    time_t t = time(0);
    return t;
}
}  // namespace advc
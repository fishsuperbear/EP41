#pragma once
#include <string>

/*****************************************************************************/
/* define globle vars for convenience, according to glog and spdlog, more    */
/* serverity, the big enum value                                             */
/*****************************************************************************/
const int UNKNOWN_L = -3; /* module not configed and first logging */
const int SCHED =
    -2; /* Deferred messages from sched code* are set to this special level */
const int DEFAULT = -1; /* default (or last) loglevel */
const int DEBUG = 0;    /* debug-level messages */
const int INFO = 1;     /* informational */
const int NOTICE = 2;   /* normal but significant condition */
const int WARNING = 3;  /* warning conditions */
const int ERROR = 4;    /* error conditions */
const int CRIT = 5;     /* critical conditions */
const int ALERT = 6;    /* action must be taken immediately */
const int FATAL = 7;    /* system is unusable */

namespace netaos {
namespace framework {

enum class LogLevel : int {
  UNKNOWN = UNKNOWN_L,
  DEBUG = DEBUG,
  INFO = INFO,
  WARNING = WARNING,
  ERROR = ERROR,
  FATAL = FATAL
};

namespace loginterface {
LogLevel get_level_enum(const std::string& level_des);
std::string get_level_desc(LogLevel level);
}  // namespace loginterface

}  // namespace framework
}  // namespace netaos
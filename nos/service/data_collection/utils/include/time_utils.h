/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: time_utils.h
 * @Date: 2023/08/15
 * @Author: cheng
 * @Desc: --
 */

#ifndef SERVICE_DATA_COLLECTION_COMMON_UTILS_TIME_UTILS_H
#define SERVICE_DATA_COLLECTION_COMMON_UTILS_TIME_UTILS_H

#include <string.h>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
namespace hozon {
namespace netaos {
namespace dc {

struct Timestamp {
  uint32_t nsec;
  uint32_t sec;
};

#define CLOCK_VIRTUAL 12
class TimeUtils {
 public:
  static std::string GET_TIMER_USE_PLANE;
  static int32_t TIME_OFFSET_SECOND;
  static void sleepWithWakeup(const long long milliseconds,
                              std::atomic_bool &wakeupFlag,
                              const long long minIntervalMs);

  static void sleepWithWakeup(const long long milliseconds, std::atomic_bool &wakeupFlag);
  static void sleep(const long long milliseconds);

  // Get data time in micro second unit
  static uint64_t getDataTimeMicroSec();
  // Get management time in micro second unit
  static uint64_t getMgmtTimeMicroSec();
  // Get data time in second unit
  static uint32_t getDataTimeSec();
  // Get management time in second unit
  static uint32_t getMgmtTimeSec();
  // Get data time macro second in macro second unit
  static uint32_t getDataTimeMicroSecInSec();
  // Get management time macro second in macro second unit
  static uint32_t getMgmtTimeMicroSecInSec();
  // Get management time stamp
  static Timestamp getMgmtTimestamp();
  // Get data time stamp
  static Timestamp getDataTimestamp();
//  // Get management time.
  static uint64_t getMgmtTime();
//  // Get data time
  static uint64_t getDataTime();
  // Convert second to yyyy/mm/dd hhmmss string.
  static std::string sec2ReadableStr(uint32_t sec);
  // Convert timestamp to yyyy/mm/dd hhmmss.000 string.
  static std::string timestamp2ReadableStr(Timestamp timestamp);

  // Convert time tick to human readable string.
  static std::string convertTime2ReadableStr(uint32_t sec);
  // Set thread name for current thread.
  static void SetThreadName(std::string name);

  // Format time into string that can be used in file name.
  static std::string formatTimeStrForFileName(time_t unix_time);
  static std::string formatTimeStrForFileName(Timestamp timestamp);

};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_COMMON_UTILS_TIME_UTILS_H

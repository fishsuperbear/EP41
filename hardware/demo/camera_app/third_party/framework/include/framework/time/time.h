#ifndef CYBER_TIME_TIME_H_
#define CYBER_TIME_TIME_H_

#include <limits>
#include <string>

#include "framework/time/duration.h"

namespace netaos {
namespace framework {

/**
 * @brief Netaos framework has builtin time type Time.
 */
class Time {
 public:
  static const Time MAX;
  static const Time MIN;
  Time() = default;
  explicit Time(uint64_t nanoseconds);
  explicit Time(int nanoseconds);
  explicit Time(double seconds);
  Time(uint32_t seconds, uint32_t nanoseconds);
  Time(const Time& other);
  Time& operator=(const Time& other);

  /**
   * @brief get the current time.
   *
   * @return return the current time.
   */
  static Time Now();
  static Time MonoTime();

  /**
   * @brief Sleep Until time.
   *
   * @param time the Time object.
   */
  static void SleepUntil(const Time& time);

  /**
   * @brief convert time to second.
   *
   * @return return a double value unit is second.
   */
  double ToSecond() const;

  /**
   * @brief convert time to microsecond (us).
   *
   * @return return a unit64_t value unit is us.
   */
  uint64_t ToMicrosecond() const;

  /**
   * @brief convert time to nanosecond.
   *
   * @return return a unit64_t value unit is nanosecond.
   */
  uint64_t ToNanosecond() const;

  /**
   * @brief convert time to a string.
   *
   * @return return a string.
   */
  std::string ToString() const;

  /**
   * @brief determine if time is 0
   *
   * @return return true if time is 0
   */
  bool IsZero() const;

  Duration operator-(const Time& rhs) const;
  Time operator+(const Duration& rhs) const;
  Time operator-(const Duration& rhs) const;
  Time& operator+=(const Duration& rhs);
  Time& operator-=(const Duration& rhs);
  bool operator==(const Time& rhs) const;
  bool operator!=(const Time& rhs) const;
  bool operator>(const Time& rhs) const;
  bool operator<(const Time& rhs) const;
  bool operator>=(const Time& rhs) const;
  bool operator<=(const Time& rhs) const;

 private:
  uint64_t nanoseconds_ = 0;
};

std::ostream& operator<<(std::ostream& os, const Time& rhs);

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TIME_TIME_H_

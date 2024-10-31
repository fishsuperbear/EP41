#include <time.h>

#include <iostream>

class TimeDiffCalc {
 public:
  TimeDiffCalc() {
    clock_gettime(CLOCK_MONOTONIC, &start);
    // std::cout << "timespec id " << ++timespec_id << " 's start sec: "
    //    << start.tv_sec << " start nsec: " << start.tv_nsec << std::endl;
  };
  ~TimeDiffCalc() {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    auto diff = calcdiff_ns(end, start);
    // std::cout << "timespec id " << ++timespec_id << " 's end sec: "
    //    << end.tv_sec << " end nsec: " << end.tv_nsec << std::endl;

    std::cout << "timespec id " << timespec_id << " 's life time "
              << diff / 1000000 << " ms seconds." << std::endl;
  };

  static uint32_t timespec_id;
  static uint64_t usec_per_sec;
  static uint64_t nsec_per_sec;

  void print_elasped() {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    auto diff = calcdiff_ns(end, start);
    // std::cout << "timespec id " << ++timespec_id << " 's end sec: "
    //    << end.tv_sec << " end nsec: " << end.tv_nsec << std::endl;

    std::cout << "timespec id " << timespec_id << " 's has elasped time "
              << diff / 1000000 << " ms seconds." << std::endl;
  }

 private:
  struct timespec start;
  static inline int64_t calcdiff(struct timespec t1, struct timespec t2) {
    int64_t diff;
    diff = usec_per_sec * (long long)((int)t1.tv_sec - (int)t2.tv_sec);
    diff += ((int)t1.tv_nsec - (int)t2.tv_nsec) / 1000;
    return diff;
  }

  static inline int64_t calcdiff_ns(struct timespec t1, struct timespec t2) {
    int64_t diff;
    diff = nsec_per_sec * (int64_t)((int)t1.tv_sec - (int)t2.tv_sec);
    diff += ((int)t1.tv_nsec - (int)t2.tv_nsec);
    return diff;
  }
};

uint32_t TimeDiffCalc::timespec_id = 0;
uint64_t TimeDiffCalc::usec_per_sec{1000000};
uint64_t TimeDiffCalc::nsec_per_sec{1000000000};

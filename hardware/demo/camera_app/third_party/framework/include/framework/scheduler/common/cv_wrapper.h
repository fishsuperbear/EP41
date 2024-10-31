#ifndef CYBER_SCHEDULER_COMMON_CV_WRAPPER_H_
#define CYBER_SCHEDULER_COMMON_CV_WRAPPER_H_

#include <condition_variable>

namespace netaos {
namespace framework {
namespace scheduler {

class CvWrapper {
 public:
  CvWrapper& operator=(const CvWrapper& other) = delete;
  std::condition_variable& Cv() { return cv_; }

 private:
  mutable std::condition_variable cv_;
};

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_COMMON_CV_WRAPPER_H_

#ifndef CYBER_SCHEDULER_POLICY_PROCESSOR_CONTEXT_H_
#define CYBER_SCHEDULER_POLICY_PROCESSOR_CONTEXT_H_

#include <limits>
#include <memory>
#include <mutex>

#include "framework/base/macros.h"
#include "framework/croutine/croutine.h"

namespace netaos {
namespace framework {
namespace scheduler {

using croutine::CRoutine;

class ProcessorContext {
 public:
  virtual void Shutdown();
  virtual std::shared_ptr<CRoutine> NextRoutine() = 0;
  virtual void Wait() = 0;

 protected:
  std::atomic<bool> stop_{false};
};

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_PROCESSOR_CONTEXT_H_

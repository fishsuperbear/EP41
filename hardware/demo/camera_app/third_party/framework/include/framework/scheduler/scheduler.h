#ifndef CYBER_SCHEDULER_SCHEDULER_H_
#define CYBER_SCHEDULER_SCHEDULER_H_

#include <unistd.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "framework/proto/choreography_conf.pb.h"

#include "framework/base/atomic_hash_map.h"
#include "framework/base/atomic_rw_lock.h"
#include "framework/common/log.h"
#include "framework/common/macros.h"
#include "framework/common/types.h"
#include "framework/croutine/croutine.h"
#include "framework/croutine/routine_factory.h"
#include "framework/scheduler/common/mutex_wrapper.h"
#include "framework/scheduler/common/pin_thread.h"

namespace netaos {
namespace framework {
namespace scheduler {

using netaos::framework::base::AtomicHashMap;
using netaos::framework::base::AtomicRWLock;
using netaos::framework::base::ReadLockGuard;
using netaos::framework::croutine::CRoutine;
using netaos::framework::croutine::RoutineFactory;
using netaos::framework::data::DataVisitorBase;
using netaos::framework::proto::InnerThread;

class Processor;
class ProcessorContext;

class Scheduler {
 public:
  virtual ~Scheduler() {}
  static Scheduler* Instance();

  bool CreateTask(const RoutineFactory& factory, const std::string& name);
  bool CreateTask(std::function<void()>&& func, const std::string& name,
                  std::shared_ptr<DataVisitorBase> visitor = nullptr);
  bool NotifyTask(uint64_t crid);

  void Shutdown();
  uint32_t TaskPoolSize() { return task_pool_size_; }

  virtual bool RemoveTask(const std::string& name) = 0;

  void ProcessLevelResourceControl();
  void SetInnerThreadAttr(const std::string& name, std::thread* thr);

  virtual bool DispatchTask(const std::shared_ptr<CRoutine>&) = 0;
  virtual bool NotifyProcessor(uint64_t crid) = 0;
  virtual bool RemoveCRoutine(uint64_t crid) = 0;

  void CheckSchedStatus();

  void SetInnerThreadConfs(
      const std::unordered_map<std::string, InnerThread>& confs) {
    inner_thr_confs_ = confs;
  }

 protected:
  Scheduler() : stop_(false) {}

  AtomicRWLock id_cr_lock_;
  AtomicHashMap<uint64_t, MutexWrapper*> id_map_mutex_;
  std::mutex cr_wl_mtx_;

  std::unordered_map<uint64_t, std::shared_ptr<CRoutine>> id_cr_;
  std::vector<std::shared_ptr<ProcessorContext>> pctxs_;
  std::vector<std::shared_ptr<Processor>> processors_;

  std::unordered_map<std::string, InnerThread> inner_thr_confs_;

  std::string process_level_cpuset_;
  uint32_t proc_num_ = 0;
  uint32_t task_pool_size_ = 0;
  std::atomic<bool> stop_;
};

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_SCHEDULER_H_

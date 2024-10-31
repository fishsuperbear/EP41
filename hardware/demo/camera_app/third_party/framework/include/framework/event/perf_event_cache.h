#ifndef CYBER_EVENT_PERF_EVENT_CACHE_H_
#define CYBER_EVENT_PERF_EVENT_CACHE_H_

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include "framework/proto/perf_conf.pb.h"

#include "framework/base/bounded_queue.h"
#include "framework/common/macros.h"
#include "framework/event/perf_event.h"

namespace netaos {
namespace framework {
namespace event {

class PerfEventCache {
 public:
  using EventBasePtr = std::shared_ptr<EventBase>;

  ~PerfEventCache();
  void AddSchedEvent(const SchedPerf event_id, const uint64_t cr_id,
                     const int proc_id, const int cr_state = -1);
  void AddTransportEvent(const TransPerf event_id, const uint64_t channel_id,
                         const uint64_t msg_seq, const uint64_t stamp = 0,
                         const std::string& adder = "-");

  std::string PerfFile() { return perf_file_; }

  void Shutdown();

 private:
  void Start();
  void Run();

  std::thread io_thread_;
  std::ofstream of_;

  bool enable_ = false;
  bool shutdown_ = false;

  proto::PerfConf perf_conf_;
  std::string perf_file_ = "";
  base::BoundedQueue<EventBasePtr> event_queue_;

  const int kFlushSize = 512;
  const uint64_t kEventQueueSize = 8192;

  DECLARE_SINGLETON(PerfEventCache)
};

}  // namespace event
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_EVENT_PERF_EVENT_CACHE_H_

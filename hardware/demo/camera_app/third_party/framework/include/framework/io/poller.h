#ifndef CYBER_IO_POLLER_H_
#define CYBER_IO_POLLER_H_

#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "framework/base/atomic_rw_lock.h"
#include "framework/common/macros.h"
#include "framework/io/poll_data.h"

namespace netaos {
namespace framework {
namespace io {

class Poller {
 public:
  using RequestPtr = std::shared_ptr<PollRequest>;
  using RequestMap = std::unordered_map<int, RequestPtr>;
  using CtrlParamMap = std::unordered_map<int, PollCtrlParam>;

  virtual ~Poller();

  void Shutdown();

  bool Register(const PollRequest& req);
  bool Unregister(const PollRequest& req);

 private:
  bool Init();
  void Clear();
  void Poll(int timeout_ms);
  void ThreadFunc();
  void HandleChanges();
  int GetTimeoutMs();
  void Notify();

  int epoll_fd_ = -1;
  std::thread thread_;
  std::atomic<bool> is_shutdown_ = {true};

  int pipe_fd_[2] = {-1, -1};
  std::mutex pipe_mutex_;

  RequestMap requests_;
  CtrlParamMap ctrl_params_;
  base::AtomicRWLock poll_data_lock_;

  const int kPollSize = 32;
  const int kPollTimeoutMs = 100;

  DECLARE_SINGLETON(Poller)
};

}  // namespace io
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_IO_POLLER_H_

#ifndef CYBER_IO_POLL_HANDLER_H_
#define CYBER_IO_POLL_HANDLER_H_

#include <atomic>
#include <memory>

#include "framework/croutine/croutine.h"
#include "framework/io/poll_data.h"

namespace netaos {
namespace framework {
namespace io {

class PollHandler {
 public:
  explicit PollHandler(int fd);
  virtual ~PollHandler() = default;

  bool Block(int timeout_ms, bool is_read);
  bool Unblock();

  int fd() const { return fd_; }
  void set_fd(int fd) { fd_ = fd; }

 private:
  bool Check(int timeout_ms);
  void Fill(int timeout_ms, bool is_read);
  void ResponseCallback(const PollResponse& rsp);

  int fd_;
  PollRequest request_;
  PollResponse response_;
  std::atomic<bool> is_read_;
  std::atomic<bool> is_blocking_;
  croutine::CRoutine* routine_;
};

}  // namespace io
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_IO_POLL_HANDLER_H_

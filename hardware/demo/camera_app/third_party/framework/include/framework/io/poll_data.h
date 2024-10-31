#ifndef CYBER_IO_POLL_DATA_H_
#define CYBER_IO_POLL_DATA_H_

#include <sys/epoll.h>

#include <cstdint>
#include <functional>

namespace netaos {
namespace framework {
namespace io {

struct PollResponse {
  explicit PollResponse(uint32_t e = 0) : events(e) {}

  uint32_t events;
};

struct PollRequest {
  int fd = -1;
  uint32_t events = 0;
  int timeout_ms = -1;
  std::function<void(const PollResponse&)> callback = nullptr;
};

struct PollCtrlParam {
  int operation;
  int fd;
  epoll_event event;
};

}  // namespace io
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_IO_POLL_DATA_H_

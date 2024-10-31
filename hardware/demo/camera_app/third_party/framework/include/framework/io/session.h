#ifndef CYBER_IO_SESSION_H_
#define CYBER_IO_SESSION_H_

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <memory>

#include "framework/io/poll_handler.h"

namespace netaos {
namespace framework {
namespace io {

class Session {
 public:
  using SessionPtr = std::shared_ptr<Session>;
  using PollHandlerPtr = std::unique_ptr<PollHandler>;

  Session();
  explicit Session(int fd);
  virtual ~Session() = default;

  int Socket(int domain, int type, int protocol);
  int Listen(int backlog);
  int Bind(const struct sockaddr *addr, socklen_t addrlen);
  SessionPtr Accept(struct sockaddr *addr, socklen_t *addrlen);
  int Connect(const struct sockaddr *addr, socklen_t addrlen);
  int Close();

  // timeout_ms < 0, keep trying until the operation is successfully
  // timeout_ms == 0, try once
  // timeout_ms > 0, keep trying while there is still time left
  ssize_t Recv(void *buf, size_t len, int flags, int timeout_ms = -1);
  ssize_t RecvFrom(void *buf, size_t len, int flags, struct sockaddr *src_addr,
                   socklen_t *addrlen, int timeout_ms = -1);

  ssize_t Send(const void *buf, size_t len, int flags, int timeout_ms = -1);
  ssize_t SendTo(const void *buf, size_t len, int flags,
                 const struct sockaddr *dest_addr, socklen_t addrlen,
                 int timeout_ms = -1);

  ssize_t Read(void *buf, size_t count, int timeout_ms = -1);
  ssize_t Write(const void *buf, size_t count, int timeout_ms = -1);

  int fd() const { return fd_; }

 private:
  void set_fd(int fd) {
    fd_ = fd;
    poll_handler_->set_fd(fd);
  }

  int fd_;
  PollHandlerPtr poll_handler_;
};

}  // namespace io
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_IO_SESSION_H_

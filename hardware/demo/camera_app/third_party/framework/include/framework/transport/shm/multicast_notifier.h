#ifndef CYBER_TRANSPORT_SHM_MULTICAST_NOTIFIER_H_
#define CYBER_TRANSPORT_SHM_MULTICAST_NOTIFIER_H_

#include <netinet/in.h>
#include <atomic>

#include "framework/common/macros.h"
#include "framework/transport/shm/notifier_base.h"

namespace netaos {
namespace framework {
namespace transport {

class MulticastNotifier : public NotifierBase {
 public:
  virtual ~MulticastNotifier();

  void Shutdown() override;
  bool Notify(const ReadableInfo& info) override;
  bool Listen(int timeout_ms, ReadableInfo* info) override;

  static const char* Type() { return "multicast"; }

 private:
  bool Init();

  int notify_fd_ = -1;
  struct sockaddr_in notify_addr_;
  int listen_fd_ = -1;
  struct sockaddr_in listen_addr_;

  std::atomic<bool> is_shutdown_ = {false};

  DECLARE_SINGLETON(MulticastNotifier)
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_SHM_MULTICAST_NOTIFIER_H_

#ifndef CYBER_TRANSPORT_SHM_NOTIFIER_BASE_H_
#define CYBER_TRANSPORT_SHM_NOTIFIER_BASE_H_

#include <memory>

#include "framework/transport/shm/readable_info.h"

namespace netaos {
namespace framework {
namespace transport {

class NotifierBase;
using NotifierPtr = NotifierBase*;

class NotifierBase {
 public:
  virtual ~NotifierBase() = default;

  virtual void Shutdown() = 0;
  virtual bool Notify(const ReadableInfo& info) = 0;
  virtual bool Listen(int timeout_ms, ReadableInfo* info) = 0;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_SHM_NOTIFIER_BASE_H_

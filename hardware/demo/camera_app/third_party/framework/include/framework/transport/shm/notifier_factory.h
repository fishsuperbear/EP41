#ifndef CYBER_TRANSPORT_SHM_NOTIFIER_FACTORY_H_
#define CYBER_TRANSPORT_SHM_NOTIFIER_FACTORY_H_

#include <memory>

#include "framework/transport/shm/notifier_base.h"

namespace netaos {
namespace framework {
namespace transport {

class NotifierFactory {
 public:
  static NotifierPtr CreateNotifier();

 private:
  static NotifierPtr CreateConditionNotifier();
  static NotifierPtr CreateMulticastNotifier();
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_SHM_NOTIFIER_FACTORY_H_

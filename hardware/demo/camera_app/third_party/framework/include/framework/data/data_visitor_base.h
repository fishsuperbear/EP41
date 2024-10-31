#ifndef CYBER_DATA_DATA_VISITOR_BASE_H_
#define CYBER_DATA_DATA_VISITOR_BASE_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "framework/common/global_data.h"
#include "framework/common/log.h"
#include "framework/data/data_notifier.h"

namespace netaos {
namespace framework {
namespace data {

class DataVisitorBase {
 public:
  DataVisitorBase() : notifier_(new Notifier()) {}

  void RegisterNotifyCallback(std::function<void()>&& callback) {
    notifier_->callback = callback;
  }

 protected:
  DataVisitorBase(const DataVisitorBase&) = delete;
  DataVisitorBase& operator=(const DataVisitorBase&) = delete;

  uint64_t next_msg_index_ = 0;
  DataNotifier* data_notifier_ = DataNotifier::Instance();
  std::shared_ptr<Notifier> notifier_;
};

}  // namespace data
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_DATA_DATA_VISITOR_BASE_H_

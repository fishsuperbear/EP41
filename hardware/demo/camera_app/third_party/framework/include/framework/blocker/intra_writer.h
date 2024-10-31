#ifndef CYBER_BLOCKER_INTRA_WRITER_H_
#define CYBER_BLOCKER_INTRA_WRITER_H_

#include <memory>

#include "framework/blocker/blocker_manager.h"
#include "framework/node/writer.h"

namespace netaos {
namespace framework {
namespace blocker {

template <typename MessageT>
class IntraWriter : public netaos::framework::Writer<MessageT> {
 public:
  using MessagePtr = std::shared_ptr<MessageT>;
  using BlockerManagerPtr = std::shared_ptr<BlockerManager>;

  explicit IntraWriter(const proto::RoleAttributes& attr);
  virtual ~IntraWriter();

  bool Init() override;
  void Shutdown() override;

  bool Write(const MessageT& msg) override;
  bool Write(const MessagePtr& msg_ptr) override;

 private:
  BlockerManagerPtr blocker_manager_;
};

template <typename MessageT>
IntraWriter<MessageT>::IntraWriter(const proto::RoleAttributes& attr)
    : Writer<MessageT>(attr) {}

template <typename MessageT>
IntraWriter<MessageT>::~IntraWriter() {
  Shutdown();
}

template <typename MessageT>
bool IntraWriter<MessageT>::Init() {
  {
    std::lock_guard<std::mutex> g(this->lock_);
    if (this->init_) {
      return true;
    }
    blocker_manager_ = BlockerManager::Instance();
    blocker_manager_->GetOrCreateBlocker<MessageT>(
        BlockerAttr(this->role_attr_.channel_name()));
    this->init_ = true;
  }
  return true;
}

template <typename MessageT>
void IntraWriter<MessageT>::Shutdown() {
  {
    std::lock_guard<std::mutex> g(this->lock_);
    if (!this->init_) {
      return;
    }
    this->init_ = false;
  }
  blocker_manager_ = nullptr;
}

template <typename MessageT>
bool IntraWriter<MessageT>::Write(const MessageT& msg) {
  if (!WriterBase::IsInit()) {
    return false;
  }
  return blocker_manager_->Publish<MessageT>(this->role_attr_.channel_name(),
                                             msg);
}

template <typename MessageT>
bool IntraWriter<MessageT>::Write(const MessagePtr& msg_ptr) {
  if (!WriterBase::IsInit()) {
    return false;
  }
  return blocker_manager_->Publish<MessageT>(this->role_attr_.channel_name(),
                                             msg_ptr);
}

}  // namespace blocker
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_BLOCKER_INTRA_WRITER_H_

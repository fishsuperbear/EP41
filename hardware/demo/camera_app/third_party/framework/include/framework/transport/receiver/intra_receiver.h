#ifndef CYBER_TRANSPORT_RECEIVER_INTRA_RECEIVER_H_
#define CYBER_TRANSPORT_RECEIVER_INTRA_RECEIVER_H_

#include "framework/common/log.h"
#include "framework/transport/dispatcher/intra_dispatcher.h"
#include "framework/transport/receiver/receiver.h"

namespace netaos {
namespace framework {
namespace transport {

template <typename M>
class IntraReceiver : public Receiver<M> {
 public:
  IntraReceiver(const RoleAttributes& attr,
                const typename Receiver<M>::MessageListener& msg_listener);
  virtual ~IntraReceiver();

  void Enable() override;
  void Disable() override;

  void Enable(const RoleAttributes& opposite_attr) override;
  void Disable(const RoleAttributes& opposite_attr) override;

 private:
  IntraDispatcherPtr dispatcher_;
};

template <typename M>
IntraReceiver<M>::IntraReceiver(
    const RoleAttributes& attr,
    const typename Receiver<M>::MessageListener& msg_listener)
    : Receiver<M>(attr, msg_listener) {
  dispatcher_ = IntraDispatcher::Instance();
}

template <typename M>
IntraReceiver<M>::~IntraReceiver() {
  Disable();
}

template <typename M>
void IntraReceiver<M>::Enable() {
  if (this->enabled_) {
    return;
  }

  dispatcher_->AddListener<M>(
      this->attr_, std::bind(&IntraReceiver<M>::OnNewMessage, this,
                             std::placeholders::_1, std::placeholders::_2));
  this->enabled_ = true;
}

template <typename M>
void IntraReceiver<M>::Disable() {
  if (!this->enabled_) {
    return;
  }

  dispatcher_->RemoveListener<M>(this->attr_);
  this->enabled_ = false;
}

template <typename M>
void IntraReceiver<M>::Enable(const RoleAttributes& opposite_attr) {
  dispatcher_->AddListener<M>(
      this->attr_, opposite_attr,
      std::bind(&IntraReceiver<M>::OnNewMessage, this, std::placeholders::_1,
                std::placeholders::_2));
}

template <typename M>
void IntraReceiver<M>::Disable(const RoleAttributes& opposite_attr) {
  dispatcher_->RemoveListener<M>(this->attr_, opposite_attr);
}

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_RECEIVER_INTRA_RECEIVER_H_
